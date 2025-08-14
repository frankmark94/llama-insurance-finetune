"""
Training utilities for LLaMA insurance fine-tuning project.

This module provides reusable functions for model training, callbacks,
and training process management.
"""

import os
import json
import torch
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from transformers import BitsAndBytesConfig


class TrainingLogger(TrainerCallback):
    """Custom callback for enhanced training logging."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.training_log = []
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def on_train_begin(self, args, state, control, **kwargs):
        """Called at the beginning of training."""
        self.logger.info(f"Training started at {datetime.now()}")
        self.logger.info(f"Total steps planned: {state.max_steps}")
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when logging."""
        if logs:
            # Add timestamp to logs
            log_entry = {
                'step': state.global_step,
                'epoch': state.epoch,
                'timestamp': datetime.now().isoformat(),
                **logs
            }
            self.training_log.append(log_entry)
            
            # Log key metrics
            if 'loss' in logs:
                self.logger.info(f"Step {state.global_step}: Loss = {logs['loss']:.4f}")
            if 'eval_loss' in logs:
                self.logger.info(f"Step {state.global_step}: Eval Loss = {logs['eval_loss']:.4f}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        self.logger.info(f"Training completed at {datetime.now()}")
        self.logger.info(f"Total steps completed: {state.global_step}")
        
        # Save training log
        log_file = self.log_dir / 'training_log.json'
        with open(log_file, 'w') as f:
            json.dump(self.training_log, f, indent=2)


def setup_model_for_training(model_name: str, 
                           lora_config: Dict,
                           quantization_config: Optional[BitsAndBytesConfig] = None) -> tuple:
    """Set up model and tokenizer for training."""
    
    print(f"Loading model: {model_name}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization
    model_kwargs = {
        'trust_remote_code': True,
        'torch_dtype': torch.float16,
        'device_map': 'auto'
    }
    
    if quantization_config:
        model_kwargs['quantization_config'] = quantization_config
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    
    # Resize embeddings if needed
    if len(tokenizer) > model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    
    # Prepare for k-bit training if quantization is used
    if quantization_config:
        model = prepare_model_for_kbit_training(model)
    
    # Set up LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        lora_dropout=lora_config['lora_dropout'],
        bias=lora_config['bias'],
        target_modules=lora_config['target_modules'],
        inference_mode=False
    )
    
    model = get_peft_model(model, peft_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    print(f"Total parameters: {total_params:,}")
    
    return model, tokenizer


def create_training_arguments(config: Dict, output_dir: Path) -> TrainingArguments:
    """Create TrainingArguments from configuration."""
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Update paths in config
    config = config.copy()
    config['output_dir'] = str(output_dir)
    config['logging_dir'] = str(output_dir / 'logs')
    
    # Create logs directory
    Path(config['logging_dir']).mkdir(exist_ok=True)
    
    return TrainingArguments(**config)


def create_trainer(model, tokenizer, training_args, train_dataset, eval_dataset=None, 
                  data_collator=None, callbacks=None) -> Trainer:
    """Create and configure Trainer."""
    
    if callbacks is None:
        callbacks = []
    
    # Add early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,
        early_stopping_threshold=0.01
    )
    callbacks.append(early_stopping)
    
    # Add training logger
    log_dir = Path(training_args.output_dir) / 'logs'
    training_logger = TrainingLogger(log_dir)
    callbacks.append(training_logger)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    return trainer


def save_model_and_tokenizer(model, tokenizer, output_dir: Path, 
                            training_args: Optional[Dict] = None,
                            lora_config: Optional[Dict] = None):
    """Save trained model, tokenizer, and metadata."""
    
    output_dir.mkdir(exist_ok=True)
    
    print(f"Saving model to {output_dir}")
    
    # Save model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save metadata
    metadata = {
        'model_type': 'LLaMA with LoRA',
        'save_timestamp': datetime.now().isoformat(),
        'training_args': training_args,
        'lora_config': lora_config
    }
    
    metadata_file = output_dir / 'training_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model and metadata saved to {output_dir}")


def calculate_training_time(trainer: Trainer) -> Dict[str, Union[int, float]]:
    """Calculate estimated training time."""
    
    train_dataloader = trainer.get_train_dataloader()
    num_batches_per_epoch = len(train_dataloader)
    num_epochs = trainer.args.num_train_epochs
    
    total_steps = num_batches_per_epoch * num_epochs
    
    # Rough estimates based on hardware
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if 'T4' in gpu_name:
            estimated_seconds_per_step = 2.5
        elif 'V100' in gpu_name:
            estimated_seconds_per_step = 1.5
        else:
            estimated_seconds_per_step = 2.0
    else:
        estimated_seconds_per_step = 10.0  # CPU is much slower
    
    estimated_total_seconds = total_steps * estimated_seconds_per_step
    
    return {
        'total_steps': total_steps,
        'batches_per_epoch': num_batches_per_epoch,
        'num_epochs': num_epochs,
        'estimated_seconds_per_step': estimated_seconds_per_step,
        'estimated_total_seconds': estimated_total_seconds,
        'estimated_hours': estimated_total_seconds / 3600
    }


def monitor_gpu_usage():
    """Monitor and report GPU usage."""
    
    if not torch.cuda.is_available():
        return {'gpu_available': False}
    
    gpu_info = {
        'gpu_available': True,
        'gpu_name': torch.cuda.get_device_name(0),
        'total_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1e9,
        'allocated_memory_gb': torch.cuda.memory_allocated(0) / 1e9,
        'reserved_memory_gb': torch.cuda.memory_reserved(0) / 1e9,
    }
    
    gpu_info['free_memory_gb'] = gpu_info['total_memory_gb'] - gpu_info['reserved_memory_gb']
    gpu_info['utilization_percent'] = (gpu_info['allocated_memory_gb'] / gpu_info['total_memory_gb']) * 100
    
    return gpu_info


def clear_gpu_memory():
    """Clear GPU memory cache."""
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("GPU memory cleared")
    else:
        print("No GPU available")


def validate_training_setup(model, tokenizer, train_dataset, training_args) -> Dict[str, bool]:
    """Validate training setup before starting."""
    
    validation_results = {
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None,
        'dataset_available': train_dataset is not None and len(train_dataset) > 0,
        'gpu_available': torch.cuda.is_available(),
        'output_dir_exists': Path(training_args.output_dir).exists(),
        'sufficient_gpu_memory': False
    }
    
    # Check GPU memory
    if validation_results['gpu_available']:
        gpu_info = monitor_gpu_usage()
        # Need at least 8GB free for training LLaMA-7B with LoRA
        validation_results['sufficient_gpu_memory'] = gpu_info['free_memory_gb'] >= 8.0
    
    # Check dataset size
    if validation_results['dataset_available']:
        dataset_size = len(train_dataset)
        validation_results['sufficient_data'] = dataset_size >= 10  # Minimum viable dataset size
    
    validation_results['ready_for_training'] = all([
        validation_results['model_loaded'],
        validation_results['tokenizer_loaded'],
        validation_results['dataset_available'],
        validation_results['output_dir_exists']
    ])
    
    return validation_results
