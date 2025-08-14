"""
Evaluation utilities for LLaMA insurance fine-tuning project.

This module provides functions for model evaluation, metrics calculation,
and performance analysis.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import json
from datetime import datetime

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import sent_tokenize, word_tokenize
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


class InsuranceModelEvaluator:
    """Comprehensive evaluator for insurance LLaMA models."""
    
    def __init__(self, model, tokenizer, generation_config: Dict):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
    
    def evaluate_task(self, examples: List[Dict], task_type: str) -> Dict:
        """Evaluate model on a specific task."""
        
        predictions = []
        references = []
        
        print(f"Evaluating {len(examples)} examples for {task_type}")
        
        for example in examples:
            # Generate prediction
            prediction = self.generate_prediction(example)
            reference = example.get('output', '')
            
            predictions.append(prediction)
            references.append(reference)
        
        # Calculate metrics based on task type
        metrics = self._calculate_task_metrics(predictions, references, task_type)
        
        return {
            'task_type': task_type,
            'num_examples': len(examples),
            'metrics': metrics,
            'predictions': predictions[:5],  # Store first 5 predictions for inspection
            'references': references[:5]
        }
    
    def generate_prediction(self, example: Dict) -> str:
        """Generate prediction for a single example."""
        
        # Create prompt
        instruction = example.get('instruction', '')
        user_input = example.get('input', '')
        
        if 'context' in example and example['context']:
            # Q&A format
            prompt = f"[INST] {instruction}\n\nContext: {example['context']}\n\nQuestion: {user_input} [/INST]"
        else:
            prompt = f"[INST] {instruction}\n\n{user_input} [/INST]"
        
        try:
            # Generate response
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **self.generation_config)
            
            # Decode response
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating prediction: {e}")
            return ""
    
    def _calculate_task_metrics(self, predictions: List[str], references: List[str], task_type: str) -> Dict:
        """Calculate metrics specific to task type."""
        
        metrics = {}
        
        if task_type == 'CLAIM_CLASSIFICATION':
            # Classification metrics
            metrics.update(self._calculate_classification_metrics(predictions, references))
        
        elif task_type in ['POLICY_SUMMARIZATION', 'FAQ_GENERATION', 'COMPLIANCE_CHECK']:
            # Text generation metrics
            metrics.update(self._calculate_text_generation_metrics(predictions, references))
        
        elif task_type == 'CONTRACT_QA':
            # Q&A specific metrics
            metrics.update(self._calculate_qa_metrics(predictions, references))
        
        return metrics
    
    def _calculate_classification_metrics(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate classification metrics."""
        
        # Extract class names from predictions and references
        pred_classes = [self._extract_class_name(pred) for pred in predictions]
        ref_classes = [self._extract_class_name(ref) for ref in references]
        
        # Calculate metrics
        accuracy = accuracy_score(ref_classes, pred_classes)
        precision, recall, f1, _ = precision_recall_fscore_support(
            ref_classes, pred_classes, average='weighted', zero_division=0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _calculate_text_generation_metrics(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate text generation metrics (ROUGE, BLEU)."""
        
        metrics = {}
        
        # ROUGE scores
        rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            for rouge_type in rouge_scores.keys():
                rouge_scores[rouge_type].append(scores[rouge_type].fmeasure)
        
        for rouge_type in rouge_scores.keys():
            metrics[f'{rouge_type}_fmeasure'] = np.mean(rouge_scores[rouge_type])
            metrics[f'{rouge_type}_precision'] = np.mean([self.rouge_scorer.score(ref, pred)[rouge_type].precision 
                                                         for pred, ref in zip(predictions, references)])
            metrics[f'{rouge_type}_recall'] = np.mean([self.rouge_scorer.score(ref, pred)[rouge_type].recall 
                                                      for pred, ref in zip(predictions, references)])
        
        # BLEU scores
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = [word_tokenize(ref.lower())]
            
            try:
                bleu_score = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=self.smoothing)
                bleu_scores.append(bleu_score)
            except:
                bleu_scores.append(0.0)
        
        metrics['bleu_score'] = np.mean(bleu_scores)
        
        return metrics
    
    def _calculate_qa_metrics(self, predictions: List[str], references: List[str]) -> Dict:
        """Calculate Q&A specific metrics."""
        
        metrics = {}
        
        # Exact match
        exact_matches = [pred.lower().strip() == ref.lower().strip() 
                        for pred, ref in zip(predictions, references)]
        metrics['exact_match'] = np.mean(exact_matches)
        
        # Token F1 (word-level F1)
        token_f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_tokens = set(word_tokenize(pred.lower()))
            ref_tokens = set(word_tokenize(ref.lower()))
            
            if not ref_tokens:
                token_f1_scores.append(0.0)
                continue
            
            common_tokens = pred_tokens & ref_tokens
            if not common_tokens:
                token_f1_scores.append(0.0)
            else:
                precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
                recall = len(common_tokens) / len(ref_tokens)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                token_f1_scores.append(f1)
        
        metrics['token_f1'] = np.mean(token_f1_scores)
        
        # Also include ROUGE scores for Q&A
        metrics.update(self._calculate_text_generation_metrics(predictions, references))
        
        return metrics
    
    def _extract_class_name(self, text: str) -> str:
        """Extract class name from classification output."""
        
        # Look for common patterns
        import re
        
        # Clean the text
        text = text.lower().strip()
        
        # Patterns to look for
        patterns = [
            r'this is an? ([\w_]+) claim',
            r'([\w_]+) claim',
            r'category: ([\w_]+)',
            r'type: ([\w_]+)',
            r'class: ([\w_]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).replace(' ', '_')
        
        # If no pattern found, return first word
        words = text.split()
        return words[0] if words else 'unknown'


def compare_models(model1_results: Dict, model2_results: Dict, 
                  model1_name: str = "Model 1", model2_name: str = "Model 2") -> pd.DataFrame:
    """Compare results from two models."""
    
    comparison_data = []
    
    # Get all task types
    all_tasks = set(model1_results.keys()) | set(model2_results.keys())
    
    for task in all_tasks:
        if task in model1_results and task in model2_results:
            m1_metrics = model1_results[task]['metrics']
            m2_metrics = model2_results[task]['metrics']
            
            # Compare common metrics
            common_metrics = set(m1_metrics.keys()) & set(m2_metrics.keys())
            
            for metric in common_metrics:
                if isinstance(m1_metrics[metric], (int, float)):
                    comparison_data.append({
                        'Task': task,
                        'Metric': metric,
                        model1_name: m1_metrics[metric],
                        model2_name: m2_metrics[metric],
                        'Difference': m2_metrics[metric] - m1_metrics[metric],
                        'Improvement': ((m2_metrics[metric] - m1_metrics[metric]) / m1_metrics[metric] * 100) if m1_metrics[metric] != 0 else 0
                    })
    
    return pd.DataFrame(comparison_data)


def calculate_overall_score(evaluation_results: Dict[str, Dict]) -> float:
    """Calculate overall performance score across all tasks."""
    
    task_scores = []
    
    for task_type, results in evaluation_results.items():
        metrics = results['metrics']
        
        # Define primary metric for each task type
        primary_metrics = {
            'CLAIM_CLASSIFICATION': 'accuracy',
            'POLICY_SUMMARIZATION': 'rouge1_fmeasure',
            'FAQ_GENERATION': 'rouge1_fmeasure',
            'COMPLIANCE_CHECK': 'rouge1_fmeasure',
            'CONTRACT_QA': 'token_f1'
        }
        
        primary_metric = primary_metrics.get(task_type, 'rouge1_fmeasure')
        
        if primary_metric in metrics:
            task_scores.append(metrics[primary_metric])
    
    return np.mean(task_scores) if task_scores else 0.0


def generate_evaluation_report(evaluation_results: Dict[str, Dict], 
                             model_name: str = "LLaMA Insurance Model") -> str:
    """Generate a comprehensive evaluation report."""
    
    overall_score = calculate_overall_score(evaluation_results)
    
    report = f"""
# {model_name} Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Overall Score:** {overall_score:.3f}

## Summary by Task

"""
    
    for task_type, results in evaluation_results.items():
        metrics = results['metrics']
        num_examples = results['num_examples']
        
        report += f"""
### {task_type}
- **Examples:** {num_examples}
- **Key Metrics:**
"""
        
        # Show most relevant metrics
        key_metrics = ['accuracy', 'f1', 'rouge1_fmeasure', 'bleu_score', 'exact_match', 'token_f1']
        
        for metric in key_metrics:
            if metric in metrics:
                report += f"  - {metric.replace('_', ' ').title()}: {metrics[metric]:.3f}\n"
        
        report += "\n"
    
    return report


def save_evaluation_results(results: Dict, output_dir: Path, 
                          model_name: str = "insurance_llama") -> None:
    """Save evaluation results to files."""
    
    output_dir.mkdir(exist_ok=True)
    
    # Save detailed results
    results_file = output_dir / f"{model_name}_evaluation_results.json"
    
    # Prepare results for JSON serialization
    json_results = {}
    for task_type, task_results in results.items():
        json_results[task_type] = {
            'task_type': task_results['task_type'],
            'num_examples': task_results['num_examples'],
            'metrics': task_results['metrics'],
            'sample_predictions': task_results.get('predictions', [])[:3],
            'sample_references': task_results.get('references', [])[:3]
        }
    
    with open(results_file, 'w') as f:
        json.dump({
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'overall_score': calculate_overall_score(results),
            'results_by_task': json_results
        }, f, indent=2)
    
    # Save summary CSV
    summary_data = []
    for task_type, task_results in results.items():
        metrics = task_results['metrics']
        row = {'Task': task_type, 'Examples': task_results['num_examples']}
        
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                row[metric.replace('_', ' ').title()] = f"{value:.3f}"
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = output_dir / f"{model_name}_evaluation_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    # Save report
    report = generate_evaluation_report(results, model_name)
    report_file = output_dir / f"{model_name}_evaluation_report.md"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"Evaluation results saved to {output_dir}")
    print(f"Files created:")
    print(f"  - {results_file.name}")
    print(f"  - {summary_file.name}")
    print(f"  - {report_file.name}")


def load_evaluation_results(results_file: Path) -> Dict:
    """Load previously saved evaluation results."""
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    return data
