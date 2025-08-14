# 🦙 LLaMA Insurance Fine-tuning Project

A comprehensive project for fine-tuning LLaMA-2-7B on insurance-specific tasks using LoRA (Low-Rank Adaptation) and Google Colab.

## 🎯 Project Overview

This project demonstrates how to fine-tune a LLaMA model for specialized insurance tasks including claim classification, policy summarization, FAQ generation, compliance checking, and contract Q&A. The implementation uses parameter-efficient fine-tuning with LoRA and is optimized for Google Colab environments.

### 🏗️ Architecture

- **Base Model**: LLaMA-2-7b-chat-hf
- **Fine-tuning Method**: LoRA (Low-Rank Adaptation)
- **Quantization**: 4-bit with BitsAndBytesConfig
- **Platform**: Google Colab (T4 GPU compatible)
- **Framework**: Transformers + PEFT

## 📋 Features

### 5 Insurance-Specific Tasks
1. **Claim Classification** - Categorize insurance claims by type
2. **Policy Summarization** - Generate clear policy summaries
3. **FAQ Generation** - Create FAQs from policy documents
4. **Compliance Check** - Identify regulatory requirements
5. **Contract Q&A** - Answer questions about insurance contracts

### Key Capabilities
- ✅ PII detection and removal
- ✅ Text cleaning and preprocessing
- ✅ Task-specific instruction tuning
- ✅ Comprehensive evaluation metrics
- ✅ Interactive inference demo
- ✅ Memory-efficient training on limited GPU

## 📂 Where to Put Your Insurance Documents

**IMPORTANT**: Place your insurance documents in the correct location for processing:

### Required Directory Structure
```
data/
├── raw/                    # 👈 PUT YOUR INSURANCE DOCUMENTS HERE
│   ├── policies/          # Insurance policy documents (.pdf, .txt, .docx)
│   ├── claims/            # Claims documents (.pdf, .txt, .docx)
│   ├── contracts/         # Contract documents (.pdf, .txt, .docx)
│   └── compliance/        # Compliance/regulatory documents (.pdf, .txt, .docx)
├── processed/             # Processed data (auto-generated)
└── sample/                # Sample data templates (provided)
```

### Document Placement Instructions

1. **Create the raw data directories**:
   ```bash
   mkdir -p data/raw/{policies,claims,contracts,compliance}
   ```

2. **Place your documents**:
   - **Insurance Policies**: `data/raw/policies/`
   - **Claims Documents**: `data/raw/claims/`  
   - **Contract Documents**: `data/raw/contracts/`
   - **Compliance/Regulatory**: `data/raw/compliance/`

3. **Supported formats**: `.pdf`, `.txt`, `.docx`, `.doc`

4. **File naming convention** (recommended):
   ```
   policy_auto_2024_001.pdf
   claim_health_2024_045.pdf
   contract_property_commercial.pdf
   compliance_hipaa_requirements.pdf
   ```

### Example Directory After Adding Your Documents
```
data/raw/
├── policies/
│   ├── auto_policy_comprehensive.pdf
│   ├── health_policy_ppo.pdf
│   └── homeowners_policy.pdf
├── claims/
│   ├── auto_claim_collision.pdf
│   ├── health_claim_emergency.pdf
│   └── property_claim_water_damage.pdf
├── contracts/
│   ├── insurance_contract_terms.pdf
│   └── service_agreement.pdf
└── compliance/
    ├── hipaa_requirements.pdf
    ├── state_regulations.pdf
    └── privacy_policy.pdf
```

**⚠️ Privacy Notice**: The system will automatically detect and remove PII (Social Security numbers, phone numbers, emails, etc.) from your documents during processing.

## 🚀 Quick Start

### 1. Environment Setup

Open `notebooks/00_colab_setup.ipynb` in Google Colab and run all cells to:
- Install dependencies
- Configure GPU settings
- Set up authentication (HuggingFace, W&B, GitHub)
- Verify environment

### 2. Data Preprocessing

Run `notebooks/01_data_preprocessing.ipynb` to:
- Remove PII from documents
- Clean and standardize text
- Create sample insurance datasets
- Validate data quality

### 3. Tokenization

Execute `notebooks/02_tokenization.ipynb` to:
- Set up LLaMA tokenizer
- Format instruction templates
- Create data collators
- Prepare training data

### 4. Fine-tuning

Run `notebooks/03_finetuning_lora.ipynb` to:
- Load and quantize base model
- Configure LoRA parameters
- Train with memory optimization
- Save fine-tuned model

### 5. Evaluation

Use `notebooks/04_evaluation.ipynb` to:
- Calculate task-specific metrics
- Generate evaluation reports
- Compare model performance
- Validate results

### 6. Interactive Demo

Try `notebooks/05_inference_demo.ipynb` to:
- Load trained model
- Test different scenarios
- Compare generation configs
- Interactive widget interface

## 📁 Project Structure

```
llama-insurance-finetune/
├── README.md                  # This file
├── requirements.txt           # Dependencies
├── .gitignore                # Git exclusions
├── config/
│   ├── lora_config.json      # LoRA hyperparameters
│   ├── training_args.json    # Training configuration
│   └── model_card.md         # Model documentation template
├── notebooks/
│   ├── 00_colab_setup.ipynb  # Environment setup
│   ├── 01_data_preprocessing.ipynb  # Data processing
│   ├── 02_tokenization.ipynb # Tokenization setup
│   ├── 03_finetuning_lora.ipynb    # Model training
│   ├── 04_evaluation.ipynb   # Model evaluation
│   └── 05_inference_demo.ipynb     # Interactive demo
├── src/
│   ├── data_utils.py          # Data processing utilities
│   ├── training_utils.py      # Training utilities
│   ├── evaluation.py          # Evaluation utilities
│   └── inference.py           # Inference utilities
├── data/
│   └── sample/               # Sample data templates
└── outputs/
    └── final_model/          # Trained model artifacts
```

## ⚙️ Configuration

### LoRA Configuration
```json
{
  "r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "bias": "none",
  "target_modules": [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
  ]
}
```

### Training Configuration
- **Batch Size**: 4 (with gradient accumulation)
- **Learning Rate**: 2e-4
- **Max Steps**: 3000
- **Warmup Steps**: 300
- **Scheduler**: Cosine with restarts
- **Precision**: FP16

## 📊 Performance Metrics

The model is evaluated on multiple metrics per task:

- **Classification**: Accuracy, Precision, Recall, F1
- **Generation**: ROUGE-1/2/L, BLEU, Token F1
- **Q&A**: Exact Match, Token F1, ROUGE scores

Expected performance after fine-tuning:
- Claim Classification: >85% accuracy
- Policy Summarization: >0.6 ROUGE-1
- FAQ Generation: >0.5 ROUGE-1
- Compliance Check: >0.6 ROUGE-1
- Contract Q&A: >0.7 Token F1

## 🛠️ Usage Examples

### Basic Inference
```python
from src.inference import load_insurance_model

# Load model
model = load_insurance_model("outputs/final_model/lora_model")

# Classify a claim
result = model.process_insurance_task(
    'CLAIM_CLASSIFICATION',
    'Vehicle collision with property damage and minor injuries'
)
print(result['response'])
```

### Batch Processing
```python
tasks = [
    {'task_type': 'CLAIM_CLASSIFICATION', 'input': 'Auto accident claim'},
    {'task_type': 'POLICY_SUMMARIZATION', 'input': 'Health policy document...'}
]

results = model.batch_process(tasks)
```

### Custom Generation Config
```python
custom_config = {
    'max_new_tokens': 256,
    'temperature': 0.1,
    'top_p': 0.9
}

result = model.process_insurance_task(
    'CONTRACT_QA',
    'What is my deductible?',
    context='Policy document...',
    custom_config=custom_config
)
```

## 🔧 Advanced Features

### Memory Optimization
- 4-bit quantization with NF4
- Gradient checkpointing
- DeepSpeed ZeRO (optional)
- Dynamic batch sizing

### Data Processing
- Automated PII removal
- Text cleaning pipeline
- Format-specific parsers
- Quality validation

### Evaluation Suite
- Task-specific metrics
- Automated reporting
- Performance comparison
- Statistical significance testing

## 🎯 Use Cases

### Insurance Companies
- **Claims Processing**: Automatic claim categorization
- **Customer Service**: Policy explanation and FAQ
- **Compliance**: Regulatory requirement identification
- **Document Review**: Contract analysis and Q&A

### Regulatory Bodies
- **Compliance Monitoring**: Automated compliance checking
- **Policy Review**: Policy language analysis
- **Risk Assessment**: Document classification

### Insurance Brokers
- **Client Education**: Policy summarization
- **Quote Comparison**: Feature extraction
- **Compliance**: Regulatory requirement identification

## 📈 Training Requirements

### Minimum Requirements
- **GPU**: NVIDIA T4 (16GB VRAM)
- **RAM**: 12GB system RAM
- **Storage**: 50GB free space
- **Time**: 2-4 hours training

### Recommended Requirements
- **GPU**: NVIDIA V100 or A100
- **RAM**: 32GB system RAM
- **Storage**: 100GB SSD
- **Time**: 1-2 hours training

## 🔒 Security & Compliance

### PII Protection
- Automatic detection of SSN, phone, email, etc.
- Configurable redaction patterns
- Validation of PII removal
- Audit logging

### Compliance Features
- HIPAA compliance considerations
- Data retention policies
- Model versioning
- Explainability features

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size or use gradient accumulation
# Enable gradient checkpointing
# Use DeepSpeed ZeRO
```

#### Slow Training
```bash
# Use mixed precision (FP16)
# Optimize data loading
# Use gradient accumulation
```

#### Poor Performance
```bash
# Increase training data
# Adjust LoRA parameters
# Fine-tune learning rate
# Use better base model
```

### Model Loading Issues
```python
# Check model path
assert model_path.exists()

# Verify GPU memory
torch.cuda.empty_cache()

# Use CPU fallback
device = "cpu" if not torch.cuda.is_available() else "cuda"
```

## 📝 Development

### Adding New Tasks
1. Define task template in `TASK_TEMPLATES`
2. Create training examples
3. Implement evaluation metrics
4. Add to inference pipeline

### Custom Data Sources
1. Implement data loader in `src/data_utils.py`
2. Add preprocessing steps
3. Validate data quality
4. Update documentation

### Model Improvements
1. Experiment with LoRA parameters
2. Try different base models
3. Implement advanced techniques
4. A/B test configurations

## 📚 Resources

### Documentation
- [LLaMA Paper](https://arxiv.org/abs/2302.13971)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

### Datasets
- Insurance policy documents
- Claims data (synthetic)
- Regulatory documents
- FAQ databases

## 🤝 Contributing

### Setup Development Environment
```bash
git clone https://github.com/your-repo/llama-insurance-finetune
cd llama-insurance-finetune
pip install -r requirements.txt
```

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings
- Write tests

### Submitting Changes
1. Fork the repository
2. Create feature branch
3. Add tests
4. Submit pull request

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

Note: LLaMA model usage requires acceptance of Meta's license agreement.

## 🙏 Acknowledgments

- Meta AI for LLaMA models
- HuggingFace for Transformers and PEFT
- Google for Colab platform
- Insurance industry for domain expertise

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check troubleshooting section
- Review documentation
- Contact maintainers

---

**⚡ Ready to fine-tune your insurance LLaMA model? Start with `notebooks/00_colab_setup.ipynb`!**