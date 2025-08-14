# LLaMA Insurance Fine-Tuning Project

## Project Overview
This project focuses on fine-tuning a smaller LLaMA model (e.g., LLaMA-2-7B) on life and health insurance data using Google Colab. The goal is to create specialized models for insurance-specific tasks while maintaining compliance with privacy regulations.

## Project Structure
```
llama-insurance-finetune/
├── data/
│   ├── raw/                          # Raw unprocessed documents
│   ├── processed/                    # Cleaned, standardized, split data
│   └── annotations/                  # Labeled data (Q&A pairs, summaries, etc.)
│
├── notebooks/
│   ├── 00_colab_setup.ipynb         # Installs, HF auth, drive mounts
│   ├── 01_data_preprocessing.ipynb  # PII stripping, cleaning, splitting
│   ├── 02_tokenization.ipynb        # Tokenizer customization + preprocessing
│   ├── 03_finetuning_lora.ipynb     # Training with PEFT / LoRA
│   ├── 04_evaluation.ipynb          # ROUGE, BLEU, F1 testing
│   └── 05_inference_demo.ipynb      # Example downstream usage
│
├── src/
│   ├── data_utils.py                # Preprocessing, tokenizing helpers
│   ├── training_utils.py            # Trainer init, callbacks, saving
│   ├── evaluation.py                # Custom metrics, comparison tools
│   └── inference.py                 # Load model and run QA / summarization
│
├── config/
│   ├── lora_config.json             # LoRA-specific params
│   ├── training_args.json           # Learning rate, batch size, etc.
│   └── model_card.md                # Model overview and audit readiness
│
├── outputs/
│   ├── checkpoints/                 # Model checkpoints
│   ├── final_model/                 # Exported HF model format
│   └── logs/                        # Training and evaluation logs
│
├── README.md
├── requirements.txt
├── .gitignore
└── CLAUDE.md
```

## Use Cases

### Primary Tasks
- **CLAIM_CLASSIFICATION**: Categorize types of claims for pre-triage
- **POLICY_SUMMARIZATION**: Summarize long legal policy language
- **FAQ_GENERATION**: Auto-generate FAQs for customer service
- **COMPLIANCE_CHECK**: Highlight key regulatory requirements
- **CONTRACT_QA**: Answer questions given policy document context

## Google Colab Setup

### Initial Configuration
1. **Enable GPU**: Edit > Notebook settings > GPU
2. **Install Dependencies**:
   ```python
   !pip install -q transformers accelerate datasets bitsandbytes peft wandb
   ```
3. **Authenticate with Hugging Face**:
   ```python
   from huggingface_hub import login
   login()
   ```

### GitHub Integration
1. **Clone Repository**:
   ```python
   !git clone https://github.com/franklinmarkley/llama-insurance-finetune.git
   %cd llama-insurance-finetune
   ```

2. **Push Changes** (requires Personal Access Token):
   ```python
   !git config --global user.email "you@example.com"
   !git config --global user.name "Your Name"
   !git add .
   !git commit -m "Updated notebook and scripts"
   !git push
   ```

## Data Processing Pipeline

### PII Removal (notebooks/01_data_preprocessing.ipynb)
- Strip SSNs, addresses, names using regex patterns
- Remove personally identifiable information while preserving structure
- Validate data integrity post-processing

### Data Standardization
- Remove headers/footers
- Correct encoding issues
- Normalize text formatting

### Dataset Split
Create train/validation/test splits in JSON format:
```json
{
  "question": "What is the coverage limit for dental procedures?",
  "context": "Policy document excerpt...",
  "answer": "The coverage limit is $1,500 annually",
  "label": "DENTAL_COVERAGE"
}
```

## Training Configuration

### LoRA Settings (config/lora_config.json)
```json
{
  "r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "bias": "none",
  "task_type": "CAUSAL_LM"
}
```

### Training Arguments (config/training_args.json)
- **Batch Size**: 4 (Colab RAM-friendly)
- **Gradient Accumulation**: 4-8 steps
- **Learning Rate**: 2e-4 to 5e-4
- **Save Steps**: Every 500-1000 steps
- **Mixed Precision**: fp16=True
- **Max Steps**: 3000-5000 depending on dataset size

### Model Selection
- **Primary**: LLaMA-2-7B-Chat
- **Alternative**: LLaMA-2-7B-hf
- Use PEFT/LoRA for efficient fine-tuning on Colab

## Evaluation Framework

### Metrics by Task
- **Classification**: Accuracy, F1-score, Precision, Recall
- **Summarization**: ROUGE-L, ROUGE-1, ROUGE-2, BLEU
- **Question Answering**: Exact Match, Token F1, BERTScore

### Testing Protocol
1. **Gold Standards**: Manually curated test sets in `data/annotations/test_set.json`
2. **Expert Review**: SME feedback on model outputs vs. human responses
3. **Cross-validation**: K-fold validation for robust evaluation

## Compliance & Governance

### Regulatory Considerations
- **HIPAA**: Health information privacy requirements
- **GLBA**: Financial privacy standards
- **Data Retention**: Policies for training data and model artifacts

### Audit Trail
Log all:
- Data source origins and preprocessing decisions
- Training configurations and hyperparameters
- Model versions and performance metrics
- Annotation changes and reviewer feedback

### Documentation Requirements
- Model card with intended use, limitations, and bias considerations
- Data statement with source descriptions and ethical considerations
- Performance benchmarks and comparison baselines

## Quick Start Workflow

1. **Setup Environment** (`00_colab_setup.ipynb`)
   - Configure GPU, install dependencies, authenticate services

2. **Process Data** (`01_data_preprocessing.ipynb`)
   - Load raw insurance documents
   - Apply PII removal and cleaning
   - Create train/val/test splits

3. **Prepare Tokenization** (`02_tokenization.ipynb`)
   - Configure LLaMA tokenizer
   - Process datasets for training format

4. **Fine-tune Model** (`03_finetuning_lora.ipynb`)
   - Load base LLaMA model with LoRA configuration
   - Train on insurance-specific tasks
   - Save checkpoints regularly

5. **Evaluate Performance** (`04_evaluation.ipynb`)
   - Run metrics on test set
   - Compare against baselines
   - Generate performance reports

6. **Deploy & Test** (`05_inference_demo.ipynb`)
   - Load fine-tuned model
   - Test on real insurance scenarios
   - Export for production use

## Best Practices

### Memory Management (Colab)
- Use gradient checkpointing for memory efficiency
- Clear GPU cache between training runs: `torch.cuda.empty_cache()`
- Monitor RAM usage and restart runtime if needed

### Version Control
- Use `.gitignore` for large files (datasets, model weights)
- Commit configuration changes frequently
- Tag important model versions

### Model Saving
- Save both LoRA adapters and merged models
- Export to Hugging Face format for easy deployment
- Backup to Google Drive for persistence

## Commands Reference

### Environment Setup
```bash
# Install core dependencies
pip install transformers accelerate datasets bitsandbytes peft

# Development tools
pip install jupyter wandb tensorboard
```

### Training Commands
```bash
# Run preprocessing
python src/data_utils.py --input data/raw --output data/processed

# Start training
python src/training_utils.py --config config/training_args.json

# Evaluate model
python src/evaluation.py --model_path outputs/final_model --test_data data/processed/test.json
```

### Testing & Validation
```bash
# Run evaluation metrics
python -m pytest src/test_evaluation.py

# Generate model card
python src/generate_model_card.py --model outputs/final_model
```

This project structure ensures reproducible, compliant, and effective fine-tuning of LLaMA models for insurance-specific applications while maintaining the flexibility to work efficiently in Google Colab environments.