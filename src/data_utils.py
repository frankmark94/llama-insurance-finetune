"""
Data utilities for LLaMA insurance fine-tuning project.

This module provides reusable functions for data preprocessing, PII removal,
text cleaning, and dataset creation.
"""

import re
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import hashlib

from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from nltk.tokenize import sent_tokenize, word_tokenize


class PIIRemover:
    """Handle PII detection and removal from insurance documents."""
    
    def __init__(self):
        """Initialize PII patterns."""
        self.patterns = {
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b',
            'phone': r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'zip_code': r'\b\d{5}(?:-\d{4})?\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'account_number': r'\b(?:account|acct|policy)\s*#?\s*\d{6,}\b',
            'date_of_birth': r'\b(?:0?[1-9]|1[0-2])[/-](?:0?[1-9]|[12]\d|3[01])[/-](?:19|20)\d{2}\b',
            'address_number': r'\b\d{1,5}\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)\b',
        }
        
        self.replacements = {
            'ssn': '[SSN]',
            'phone': '[PHONE]',
            'email': '[EMAIL]',
            'zip_code': '[ZIP]',
            'credit_card': '[CARD_NUMBER]',
            'account_number': '[ACCOUNT_NUMBER]',
            'date_of_birth': '[DATE_OF_BIRTH]',
            'address_number': '[ADDRESS]',
        }
    
    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text and return findings."""
        findings = {}
        
        for pii_type, pattern in self.patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                findings[pii_type] = matches
        
        return findings
    
    def remove_pii(self, text: str) -> Tuple[str, Dict[str, int]]:
        """Remove PII from text and return cleaned text with removal stats."""
        cleaned_text = text
        removal_stats = {}
        
        for pii_type, replacement in self.replacements.items():
            pattern = self.patterns[pii_type]
            matches = re.findall(pattern, cleaned_text, re.IGNORECASE)
            removal_stats[pii_type] = len(matches)
            cleaned_text = re.sub(pattern, replacement, cleaned_text, flags=re.IGNORECASE)
        
        return cleaned_text, removal_stats
    
    def validate_pii_removal(self, text: str) -> bool:
        """Validate that no obvious PII remains in text."""
        findings = self.detect_pii(text)
        return len(findings) == 0


class TextCleaner:
    """Handle text cleaning and standardization."""
    
    def __init__(self):
        """Initialize text cleaner."""
        pass
    
    def clean_text(self, text: str) -> str:
        """Clean and standardize text."""
        if not isinstance(text, str):
            return ""
        
        # Remove common document artifacts
        text = self._remove_headers_footers(text)
        
        # Fix encoding issues
        text = self._fix_encoding(text)
        
        # Normalize whitespace
        text = self._normalize_whitespace(text)
        
        # Remove excessive formatting
        text = self._remove_excessive_formatting(text)
        
        return text.strip()
    
    def _remove_headers_footers(self, text: str) -> str:
        """Remove common headers and footers."""
        # Remove page numbers and headers
        text = re.sub(r'^Page \d+.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^.*Page \d+ of \d+.*$', '', text, flags=re.MULTILINE)
        
        # Remove common footer patterns
        text = re.sub(r'^.*Confidential.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'^.*Copyright.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        return text
    
    def _fix_encoding(self, text: str) -> str:
        """Fix common encoding issues."""
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€"': '-',
            'â€"': '—',
            'Â': '',
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace and line breaks."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Replace multiple line breaks with double line break
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        return text
    
    def _remove_excessive_formatting(self, text: str) -> str:
        """Remove excessive formatting characters."""
        # Remove excessive punctuation
        text = re.sub(r'[.]{3,}', '...', text)
        text = re.sub(r'[-]{3,}', '---', text)
        text = re.sub(r'[=]{3,}', '===', text)
        
        # Remove form field artifacts
        text = re.sub(r'_{3,}', '[FIELD]', text)
        
        return text
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """Extract common insurance document sections."""
        sections = {}
        
        # Common section headers
        section_patterns = {
            'coverage': r'(coverage|benefits?)\s*:?(.{0,2000}?)(?=\n[A-Z][A-Za-z\s]{10,}:|$)',
            'exclusions': r'(exclusions?)\s*:?(.{0,2000}?)(?=\n[A-Z][A-Za-z\s]{10,}:|$)',
            'deductible': r'(deductible|copay)\s*:?(.{0,500}?)(?=\n[A-Z][A-Za-z\s]{10,}:|$)',
            'premium': r'(premium|cost|price)\s*:?(.{0,500}?)(?=\n[A-Z][A-Za-z\s]{10,}:|$)',
            'terms': r'(terms?\s+and\s+conditions?)\s*:?(.{0,2000}?)(?=\n[A-Z][A-Za-z\s]{10,}:|$)',
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section_name] = match.group(2).strip()
        
        return sections


class DatasetCreator:
    """Create task-specific datasets from processed documents."""
    
    def __init__(self):
        """Initialize dataset creator."""
        self.task_types = {
            'CLAIM_CLASSIFICATION': 'Categorize insurance claims',
            'POLICY_SUMMARIZATION': 'Summarize policy documents',
            'FAQ_GENERATION': 'Generate FAQs from policies',
            'COMPLIANCE_CHECK': 'Identify compliance requirements',
            'CONTRACT_QA': 'Answer questions about contracts'
        }
    
    def create_classification_example(self, doc: Dict) -> Dict:
        """Create classification training example."""
        return {
            'instruction': 'Classify this insurance claim into the appropriate category.',
            'input': doc['content'],
            'output': f"This is a {doc['type']} claim.",
            'task_type': 'CLAIM_CLASSIFICATION',
            'doc_id': doc['id']
        }
    
    def create_summarization_example(self, doc: Dict, sections: Dict) -> Dict:
        """Create summarization training example."""
        summary = self._generate_summary(doc['content'], sections)
        
        return {
            'instruction': 'Summarize the following insurance policy document.',
            'input': doc['content'],
            'output': summary,
            'task_type': 'POLICY_SUMMARIZATION',
            'doc_id': doc['id']
        }
    
    def create_qa_examples(self, doc: Dict, sections: Dict) -> List[Dict]:
        """Create Q&A training examples."""
        qa_pairs = self._generate_qa_pairs(doc['content'], sections, doc['type'])
        examples = []
        
        for qa in qa_pairs:
            examples.append({
                'context': doc['content'],
                'question': qa['question'],
                'answer': qa['answer'],
                'instruction': 'Answer the question based on the insurance document.',
                'input': f"Context: {doc['content']}\n\nQuestion: {qa['question']}",
                'output': qa['answer'],
                'task_type': 'CONTRACT_QA',
                'doc_id': doc['id']
            })
        
        return examples
    
    def create_faq_example(self, doc: Dict, sections: Dict) -> Dict:
        """Create FAQ generation example."""
        faqs = self._generate_faqs(doc['content'], sections)
        
        return {
            'instruction': 'Generate frequently asked questions for this insurance document.',
            'input': doc['content'],
            'output': self._format_faqs(faqs),
            'task_type': 'FAQ_GENERATION',
            'doc_id': doc['id']
        }
    
    def create_compliance_example(self, doc: Dict) -> Dict:
        """Create compliance checking example."""
        compliance_items = self._extract_compliance_items(doc['content'])
        
        return {
            'instruction': 'Identify compliance requirements in this insurance document.',
            'input': doc['content'],
            'output': self._format_compliance_items(compliance_items),
            'task_type': 'COMPLIANCE_CHECK',
            'doc_id': doc['id']
        }
    
    def _generate_summary(self, content: str, sections: Dict[str, str]) -> str:
        """Generate a summary of the insurance document."""
        if sections:
            # Use sections to create structured summary
            summary_parts = []
            
            if 'coverage' in sections:
                summary_parts.append(f"Coverage: {sections['coverage'][:200]}...")
            
            if 'deductible' in sections:
                summary_parts.append(f"Deductible: {sections['deductible'][:100]}...")
            
            if 'premium' in sections:
                summary_parts.append(f"Premium: {sections['premium'][:100]}...")
            
            return " ".join(summary_parts)
        else:
            # Simple truncation summary
            sentences = sent_tokenize(content)
            return " ".join(sentences[:3]) if len(sentences) >= 3 else content[:512]
    
    def _generate_qa_pairs(self, content: str, sections: Dict[str, str], doc_type: str) -> List[Dict[str, str]]:
        """Generate question-answer pairs from document."""
        qa_pairs = []
        
        # Common insurance questions based on document type and sections
        if doc_type == 'health_policy':
            qa_pairs.extend([
                {"question": "What is the annual deductible?", "answer": sections.get('deductible', 'Deductible information not specified.')},
                {"question": "What does this policy cover?", "answer": sections.get('coverage', 'Coverage details not specified.')},
                {"question": "What is excluded from coverage?", "answer": sections.get('exclusions', 'Exclusions not specified.')}
            ])
        elif doc_type == 'auto_claim':
            qa_pairs.extend([
                {"question": "What type of claim is this?", "answer": f"This is a {doc_type.replace('_', ' ')} claim."},
                {"question": "What was the outcome of this claim?", "answer": "Claim details are provided in the document."}
            ])
        elif doc_type == 'life_policy':
            qa_pairs.extend([
                {"question": "What is the coverage amount?", "answer": sections.get('coverage', 'Coverage amount not specified.')},
                {"question": "What are the premium terms?", "answer": sections.get('premium', 'Premium terms not specified.')}
            ])
        
        # Add a general question
        qa_pairs.append({
            "question": "What is this document about?",
            "answer": f"This document is about {doc_type.replace('_', ' ')} and contains insurance-related information."
        })
        
        return qa_pairs
    
    def _generate_faqs(self, content: str, sections: Dict[str, str]) -> List[Dict[str, str]]:
        """Generate FAQs from document content."""
        faqs = [
            {
                "question": "What information does this document contain?",
                "answer": "This document contains insurance policy or claim information."
            },
            {
                "question": "How can I understand my coverage?",
                "answer": "Review the coverage section for details about what is included in your policy."
            }
        ]
        
        if 'deductible' in sections:
            faqs.append({
                "question": "What is my deductible?",
                "answer": "Your deductible information is detailed in the policy terms."
            })
        
        return faqs
    
    def _extract_compliance_items(self, content: str) -> List[str]:
        """Extract compliance-related items from content."""
        compliance_keywords = [
            'HIPAA', 'privacy', 'regulation', 'compliance', 'state law', 
            'federal law', 'requirement', 'mandatory', 'must', 'shall'
        ]
        
        compliance_items = []
        sentences = sent_tokenize(content)
        
        for sentence in sentences:
            if any(keyword.lower() in sentence.lower() for keyword in compliance_keywords):
                compliance_items.append(sentence.strip())
        
        return compliance_items[:5]  # Limit to top 5 compliance items
    
    def _format_faqs(self, faqs: List[Dict[str, str]]) -> str:
        """Format FAQs as text."""
        formatted = []
        for i, faq in enumerate(faqs, 1):
            formatted.append(f"Q{i}: {faq['question']}\nA{i}: {faq['answer']}")
        return "\n\n".join(formatted)
    
    def _format_compliance_items(self, items: List[str]) -> str:
        """Format compliance items as text."""
        return "\n".join(f"- {item}" for item in items)


def create_train_test_splits(examples: List[Dict], 
                           train_ratio: float = 0.7,
                           val_ratio: float = 0.15,
                           test_ratio: float = 0.15,
                           random_state: int = 42) -> Dict[str, List[Dict]]:
    """Create train/validation/test splits."""
    
    if len(examples) < 3:
        # Too few examples, use all for training
        return {
            'train': examples,
            'validation': examples[:1] if examples else [],
            'test': examples[:1] if examples else []
        }
    
    # Split the data
    train_examples, temp_examples = train_test_split(
        examples, 
        test_size=(val_ratio + test_ratio),
        random_state=random_state,
        shuffle=True
    )
    
    if len(temp_examples) >= 2:
        val_examples, test_examples = train_test_split(
            temp_examples,
            test_size=test_ratio/(val_ratio + test_ratio),
            random_state=random_state,
            shuffle=True
        )
    else:
        val_examples = temp_examples[:1] if temp_examples else []
        test_examples = temp_examples[1:] if len(temp_examples) > 1 else temp_examples[:1]
    
    return {
        'train': train_examples,
        'validation': val_examples,
        'test': test_examples
    }


def save_processed_datasets(data_splits: Dict[str, List[Dict]], 
                          output_dir: Path,
                          metadata: Optional[Dict] = None) -> None:
    """Save processed datasets to disk."""
    
    output_dir.mkdir(exist_ok=True)
    
    # Save each split
    for split_name, examples in data_splits.items():
        # Save as JSON
        json_file = output_dir / f"{split_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(examples, f, indent=2, ensure_ascii=False)
        
        # Save as HuggingFace dataset if we have examples
        if examples:
            dataset = Dataset.from_list(examples)
            hf_dir = output_dir / f"{split_name}_hf"
            dataset.save_to_disk(hf_dir)
    
    # Save metadata
    if metadata is None:
        metadata = {
            'processing_date': datetime.now().isoformat(),
            'total_examples': sum(len(examples) for examples in data_splits.values()),
            'splits': {split: len(examples) for split, examples in data_splits.items()}
        }
    
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_processed_datasets(data_dir: Path) -> Optional[Dict[str, Dataset]]:
    """Load processed datasets from disk."""
    
    datasets = {}
    
    for split in ['train', 'validation', 'test']:
        json_file = data_dir / f"{split}.json"
        hf_dir = data_dir / f"{split}_hf"
        
        if hf_dir.exists():
            try:
                dataset = Dataset.load_from_disk(hf_dir)
                datasets[split] = dataset
            except:
                if json_file.exists():
                    dataset = Dataset.from_json(str(json_file))
                    datasets[split] = dataset
        elif json_file.exists():
            dataset = Dataset.from_json(str(json_file))
            datasets[split] = dataset
    
    return datasets if datasets else None


def validate_dataset_quality(examples: List[Dict]) -> Dict[str, any]:
    """Validate dataset quality and return statistics."""
    
    if not examples:
        return {'valid': False, 'message': 'No examples provided'}
    
    stats = {
        'total_examples': len(examples),
        'valid_examples': 0,
        'task_distribution': {},
        'avg_input_length': 0,
        'avg_output_length': 0,
        'missing_fields': [],
        'warnings': []
    }
    
    required_fields = ['instruction', 'input', 'output', 'task_type']
    input_lengths = []
    output_lengths = []
    
    for example in examples:
        # Check required fields
        missing = [field for field in required_fields if field not in example or not example[field]]
        if not missing:
            stats['valid_examples'] += 1
            
            # Track task distribution
            task_type = example['task_type']
            stats['task_distribution'][task_type] = stats['task_distribution'].get(task_type, 0) + 1
            
            # Track lengths
            input_lengths.append(len(example['input']))
            output_lengths.append(len(example['output']))
        else:
            stats['missing_fields'].extend(missing)
    
    # Calculate averages
    if input_lengths:
        stats['avg_input_length'] = np.mean(input_lengths)
        stats['avg_output_length'] = np.mean(output_lengths)
        stats['max_input_length'] = max(input_lengths)
        stats['max_output_length'] = max(output_lengths)
    
    # Add warnings
    if stats['valid_examples'] < stats['total_examples'] * 0.9:
        stats['warnings'].append("More than 10% of examples have missing fields")
    
    if stats['avg_input_length'] > 4000:
        stats['warnings'].append("Average input length is very long (>4000 chars)")
    
    stats['valid'] = stats['valid_examples'] > 0
    
    return stats