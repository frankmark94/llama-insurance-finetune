"""
Inference utilities for LLaMA insurance fine-tuned models.

This module provides utilities for loading models, generating responses,
and handling insurance-specific inference tasks.
"""

import torch
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel


class InsuranceLLaMAInference:
    """Inference wrapper for insurance-specific LLaMA model."""
    
    def __init__(self, model_path: Union[str, Path], base_model_name: str):
        self.model_path = Path(model_path)
        self.base_model_name = base_model_name
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Default generation configs
        self.generation_configs = {
            'creative': {
                'max_new_tokens': 512,
                'temperature': 0.8,
                'top_p': 0.9,
                'top_k': 50,
                'do_sample': True,
                'repetition_penalty': 1.1
            },
            'factual': {
                'max_new_tokens': 256,
                'temperature': 0.3,
                'top_p': 0.8,
                'top_k': 40,
                'do_sample': True,
                'repetition_penalty': 1.05
            },
            'precise': {
                'max_new_tokens': 128,
                'temperature': 0.1,
                'top_p': 0.7,
                'top_k': 20,
                'do_sample': True,
                'repetition_penalty': 1.02
            }
        }
        
        # Task-specific templates
        self.task_templates = {
            'CLAIM_CLASSIFICATION': {
                'instruction': 'Classify this insurance claim into the appropriate category:',
                'config': 'precise'
            },
            'POLICY_SUMMARIZATION': {
                'instruction': 'Summarize the key points of this insurance policy:',
                'config': 'factual'
            },
            'FAQ_GENERATION': {
                'instruction': 'Generate frequently asked questions based on this insurance information:',
                'config': 'creative'
            },
            'COMPLIANCE_CHECK': {
                'instruction': 'Identify compliance requirements in this insurance document:',
                'config': 'factual'
            },
            'CONTRACT_QA': {
                'instruction': 'Answer this question based on the insurance contract:',
                'config': 'precise'
            }
        }
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer."""
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        print(f"Loading model from {self.model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Load base model
        print(f"Loading base model: {self.base_model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA adapters
        print("Loading LoRA adapters...")
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Update generation configs with tokenizer info
        for config in self.generation_configs.values():
            config['pad_token_id'] = self.tokenizer.pad_token_id
            config['eos_token_id'] = self.tokenizer.eos_token_id
        
        print("✅ Model loaded successfully!")
    
    def generate_response(self, prompt: str, generation_config: Optional[Dict] = None) -> str:
        """Generate response from the model."""
        
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        if generation_config is None:
            generation_config = self.generation_configs['factual']
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=2048
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config,
                    use_cache=True
                )
            
            # Decode response (remove input prompt)
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Error generating response. Please try again."
    
    def format_insurance_prompt(self, task_type: str, user_input: str, context: str = "") -> str:
        """Format input into proper instruction prompt for insurance tasks."""
        
        if task_type not in self.task_templates:
            raise ValueError(f"Unknown task type: {task_type}. Available: {list(self.task_templates.keys())}")
        
        template = self.task_templates[task_type]
        instruction = template['instruction']
        
        if task_type == 'CONTRACT_QA' and context:
            full_input = f"Context: {context}\n\nQuestion: {user_input}"
        else:
            full_input = user_input
        
        prompt = f"[INST] {instruction}\n\n{full_input} [/INST]"
        return prompt
    
    def process_insurance_task(self, task_type: str, user_input: str, 
                             context: str = "", custom_config: Optional[Dict] = None) -> Dict:
        """Process a complete insurance task."""
        
        # Format prompt
        prompt = self.format_insurance_prompt(task_type, user_input, context)
        
        # Get generation config
        if custom_config:
            generation_config = custom_config
            config_name = 'custom'
        else:
            config_name = self.task_templates[task_type]['config']
            generation_config = self.generation_configs[config_name]
        
        # Generate response
        response = self.generate_response(prompt, generation_config)
        
        return {
            'task_type': task_type,
            'user_input': user_input,
            'context': context,
            'prompt': prompt,
            'response': response,
            'config_used': config_name,
            'timestamp': datetime.now().isoformat()
        }
    
    def batch_process(self, tasks: List[Dict]) -> List[Dict]:
        """Process multiple tasks in batch."""
        
        results = []
        
        for task in tasks:
            task_type = task.get('task_type')
            user_input = task.get('input', '')
            context = task.get('context', '')
            custom_config = task.get('config')
            
            try:
                result = self.process_insurance_task(task_type, user_input, context, custom_config)
                results.append(result)
            except Exception as e:
                results.append({
                    'task_type': task_type,
                    'user_input': user_input,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                })
        
        return results
    
    def get_available_tasks(self) -> List[str]:
        """Get list of available task types."""
        return list(self.task_templates.keys())
    
    def get_generation_configs(self) -> Dict[str, Dict]:
        """Get available generation configurations."""
        return self.generation_configs.copy()
    
    def update_generation_config(self, config_name: str, config: Dict):
        """Update or add a generation configuration."""
        # Ensure required keys are present
        config['pad_token_id'] = self.tokenizer.pad_token_id if self.tokenizer else None
        config['eos_token_id'] = self.tokenizer.eos_token_id if self.tokenizer else None
        
        self.generation_configs[config_name] = config
        print(f"Updated generation config '{config_name}'")
    
    def save_session(self, results: List[Dict], output_file: Path):
        """Save inference session results."""
        
        session_data = {
            'model_path': str(self.model_path),
            'base_model_name': self.base_model_name,
            'session_timestamp': datetime.now().isoformat(),
            'num_tasks': len(results),
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"Session saved to {output_file}")
    
    def load_session(self, session_file: Path) -> List[Dict]:
        """Load previous inference session."""
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        return session_data.get('results', [])
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        
        if self.model is None:
            return {'status': 'not_loaded'}
        
        return {
            'status': 'loaded',
            'model_path': str(self.model_path),
            'base_model_name': self.base_model_name,
            'device': str(self.device),
            'vocab_size': len(self.tokenizer) if self.tokenizer else None,
            'available_tasks': self.get_available_tasks(),
            'generation_configs': list(self.generation_configs.keys())
        }


def load_insurance_model(model_path: Union[str, Path], 
                        base_model_name: str = "meta-llama/Llama-2-7b-chat-hf") -> InsuranceLLaMAInference:
    """Convenience function to load insurance model."""
    
    model = InsuranceLLaMAInference(model_path, base_model_name)
    model.load_model()
    return model


def create_sample_tasks() -> List[Dict]:
    """Create sample tasks for testing."""
    
    return [
        {
            'task_type': 'CLAIM_CLASSIFICATION',
            'input': 'Vehicle collision on Highway 101. Front-end damage, no injuries. Police report filed.'
        },
        {
            'task_type': 'POLICY_SUMMARIZATION',
            'input': '''Auto Insurance Policy: Comprehensive coverage with $500 deductible. 
            Bodily injury liability $100,000 per person. Property damage $50,000. 
            Monthly premium $120. Includes roadside assistance.'''
        },
        {
            'task_type': 'CONTRACT_QA',
            'input': 'What is my deductible for collision coverage?',
            'context': '''Auto Insurance Policy Section 4: Collision Coverage
            We will pay for damage to your vehicle from collision with another object. 
            Your collision deductible is $500 per occurrence.'''
        },
        {
            'task_type': 'FAQ_GENERATION',
            'input': '''Health Insurance: PPO plan with $2,000 annual deductible. 
            80% coverage after deductible. Preventive care covered 100%. 
            Prescription benefits: $10 generic, $30 brand name.'''
        },
        {
            'task_type': 'COMPLIANCE_CHECK',
            'input': '''Insurance operations must comply with state regulations and HIPAA requirements. 
            All marketing materials require state approval. Claims must be processed fairly 
            and promptly according to state laws.'''
        }
    ]


def benchmark_model_speed(model: InsuranceLLaMAInference, num_samples: int = 5) -> Dict:
    """Benchmark model inference speed."""
    
    import time
    
    sample_tasks = create_sample_tasks()[:num_samples]
    
    times = []
    token_counts = []
    
    for task in sample_tasks:
        start_time = time.time()
        
        result = model.process_insurance_task(
            task['task_type'], 
            task['input'], 
            task.get('context', '')
        )
        
        end_time = time.time()
        
        # Count tokens in response
        response_tokens = len(model.tokenizer.encode(result['response']))
        
        times.append(end_time - start_time)
        token_counts.append(response_tokens)
    
    return {
        'num_samples': num_samples,
        'avg_time_seconds': sum(times) / len(times),
        'total_time_seconds': sum(times),
        'avg_tokens_generated': sum(token_counts) / len(token_counts),
        'tokens_per_second': sum(token_counts) / sum(times),
        'times': times,
        'token_counts': token_counts
    }


class InferenceSession:
    """Manage an inference session with history and context."""
    
    def __init__(self, model: InsuranceLLaMAInference):
        self.model = model
        self.history = []
        self.session_start = datetime.now()
    
    def process_task(self, task_type: str, user_input: str, context: str = "") -> Dict:
        """Process task and add to history."""
        
        result = self.model.process_insurance_task(task_type, user_input, context)
        self.history.append(result)
        return result
    
    def get_history(self) -> List[Dict]:
        """Get session history."""
        return self.history.copy()
    
    def clear_history(self):
        """Clear session history."""
        self.history = []
    
    def get_session_stats(self) -> Dict:
        """Get session statistics."""
        
        if not self.history:
            return {'num_tasks': 0}
        
        task_types = [task['task_type'] for task in self.history]
        task_distribution = {}
        for task_type in task_types:
            task_distribution[task_type] = task_distribution.get(task_type, 0) + 1
        
        return {
            'num_tasks': len(self.history),
            'session_duration': (datetime.now() - self.session_start).total_seconds(),
            'task_distribution': task_distribution,
            'avg_response_length': sum(len(task['response']) for task in self.history) / len(self.history)
        }
    
    def save_session(self, output_file: Path):
        """Save entire session."""
        
        session_data = {
            'session_start': self.session_start.isoformat(),
            'session_end': datetime.now().isoformat(),
            'model_info': self.model.get_model_info(),
            'stats': self.get_session_stats(),
            'history': self.history
        }
        
        with open(output_file, 'w') as f:
            json.dump(session_data, f, indent=2)
        
        print(f"Session saved to {output_file}")
