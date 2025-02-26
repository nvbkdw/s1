import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import transformers
import trl
import ray
from ray import train
from ray.train import ScalingConfig
from ray.train.torch import TorchTrainer
from ray.runtime_env import RuntimeEnv


@dataclass
class TrainingConfig:
    model_name: str = field(default="Qwen/Qwen2.5-32B-Instruct")
    block_size: int = field(default=1024)
    wandb_project: Optional[str] = field(default="s1")
    wandb_entity: Optional[str] = field(default="hashimoto-group")
    train_file_path: Optional[str] = field(default='s1/s1K_tokenized')
    dagger: bool = field(default=False)

    def __post_init__(self):
        os.environ['WANDB_PROJECT'] = self.wandb_project
        os.environ['WANDB_ENTITY'] = self.wandb_entity


def llm_train(config, args=None):
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from datasets import load_dataset, concatenate_datasets, DatasetDict
    import transformers
    from ray.train.huggingface.transformers import (
        RayTrainReportCallback,
        prepare_trainer,
    )
    import trl
    from pytorch_memlab import MemReporter
    from torch.profiler import profile, schedule, tensorboard_trace_handler
    
    args_dict = asdict(args)
    training_args_dict = {}
    for k, v in args_dict.items():
        if not k.startswith('_'):
            training_args_dict[k] = v

    training_args = trl.SFTConfig(**training_args_dict)
    
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    logging.info("world size: ", world_size)
    logging.info("local rank: ", local_rank)
    
    dist.barrier()
    logging.info("Distributed training synchronization done.")
    
    # loading model
    kwargs = {
        "low_cpu_mem_usage": True,
        "device_map": "auto", 
        "torch_dtype": "auto",
        "attn_implementation": "flash_attention_2", 
        "use_cache": False
    }
    logging.info("Loading model on rank: ", local_rank)
    model = transformers.AutoModelForCausalLM.from_pretrained(config.model_name, **kwargs)

    logging.info("Loading dataset ...")
    dataset = load_dataset(config.train_file_path)
    train_dataset = dataset['train'].select(range(8))
    eval_dataset = (dataset['test'] if 'test' in dataset else dataset['train']).select(range(4))

    # setting up trainer
    tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name, use_fast=True)
    if "Llama" in config.model_name:
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|reserved_special_token_5|>"
    elif "Qwen" in config.model_name:
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
        # Use a token that is never used
        tokenizer.pad_token = "<|fim_pad|>"

    # Only compute loss over assistant responses
    # Verified that it precisely starts where the thinking tokens start and ends with the first pad token
    # via labels being set to -100
    collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
        mlm=False
    )
    training_args.dataset_text_field = 'text'
    training_args.max_seq_length = config.block_size
    training_args.report_to = [] 
    # training_args.local_rank = local_rank
    # reporter = MemReporter(model)
    # reporter.report()
    
    tracing_schedule = schedule(skip_first=5, wait=5, warmup=2, active=2, repeat=1)
    trace_handler = tensorboard_trace_handler(dir_name='/mnt/workspace/ryan/trace', use_gzip=True)
    from torch.profiler import ProfilerActivity
    from accelerate import ProfileKwargs
    
    logging.info("Distributed mode: ", args.distributed_state, args.parallel_mode)
    
    
    # define custom trainer
    class ProfilerTrainer(trl.SFTTrainer):
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.profiler = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(
                    wait=0, warmup=0, active=2,
                ),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name='/mnt/workspace/ryan/trace', use_gzip=True),
                record_shapes=True,
                profile_memory=True,
                with_flops=True,
                with_stack=True
            )

        def training_step(
            self, model, inputs, num_items_in_batch=None
        ) -> torch.Tensor:
            with self.profiler as prof:
                loss = super().training_step(model, inputs)
                prof.step()
            return loss

        def on_train_end(self):
            self.profiler.stop()
            print(self.profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=10))
            super().on_train_end()
        
        
    trainer = ProfilerTrainer(
        model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator
    )
    
    # [2] Report Metrics and Checkpoints to Ray Train
    # ===============================================
    # callback = RayTrainReportCallback()
    # trainer.add_callback(callback)

    # [3] Prepare Transformers Trainer
    # ================================
    trainer = prepare_trainer(trainer)
    dist.barrier()

    logging.info("Starting training ...")
    trainer.train()

    dist.barrier()
    logging.info("Distributed training completed.")
        
    # trainer.save_model(output_dir=training_args.output_dir)
    # tokenizer.save_pretrained(training_args.output_dir)
    trainer.accelerator.wait_for_everyone()
    logging.info("Training completed.")


def main():
    # parsing input
    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses()
    log_config = {**asdict(config), **asdict(args)}
    logging.info(f"Training config: {log_config}, \n Training args: {args}")
    
    runtime_env = RuntimeEnv(
        pip=['pytorch_memlab'],
        env_vars={
            'NCCL_SOCKET_IFNAME': 'eth0',
            'NCCL_IB_HCA': 'mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1',
            # 'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
            'NCCL_IB_GID_INDEX': '3',
            # additional python modules
            # 'PYTHONPATH': '/mnt/workspace/ryan/llm-profiler/py',
            'HF_HOME': '/mnt/workspace/ryan/.cache/huggingface',
            'WANDB_DISABLED': 'true',
        }
    )
    
    # ray.init(address="ray://localhost:10001", runtime_env=runtime_env)
    ray.init(runtime_env=runtime_env)
    scaling_config = ScalingConfig(
        num_workers=32,
        use_gpu=True,   
        resources_per_worker={"CPU": 4, "GPU": 1},
    )
    run_config = train.RunConfig(
        storage_path='~/ray_results',
    ) 
    
    def train_fn():
        llm_train(config, args)
        
    ray_trainer = TorchTrainer(
        train_fn,
        scaling_config=scaling_config,
        run_config=run_config,
    )
    ray_trainer.fit()
   
    
if __name__ == "__main__":
    main()
