# 📋 Data Organization Guide

This guide explains how to organize your insurance documents for the LLaMA fine-tuning project.

## 🎯 Quick Setup

Run this command to create all required directories:

```bash
mkdir -p data/raw/{policies,claims,contracts,compliance}
mkdir -p data/processed
```

## 📂 Directory Structure Explained

### `data/raw/` - Your Original Documents
This is where you place your original insurance documents:

- **`policies/`** - Insurance policy documents
  - Auto insurance policies
  - Health insurance policies  
  - Life insurance policies
  - Property/homeowners policies
  - Commercial insurance policies

- **`claims/`** - Claims-related documents
  - Claim forms and applications
  - Claim settlements
  - Claim denials and appeals
  - Accident reports
  - Medical reports for claims

- **`contracts/`** - Legal contracts and agreements
  - Insurance contracts
  - Service agreements
  - Reinsurance agreements
  - Agent/broker agreements

- **`compliance/`** - Regulatory and compliance documents
  - State insurance regulations
  - Federal compliance requirements
  - HIPAA documentation
  - Privacy policies
  - Audit reports

### `data/processed/` - Auto-Generated
This directory is automatically created during processing and contains:
- Cleaned documents with PII removed
- Tokenized datasets
- Training/validation/test splits

### `data/sample/` - Example Templates
Contains sample data for each task type (already provided).

## 📄 Supported File Formats

- **PDF**: `.pdf` (most common)
- **Text**: `.txt` 
- **Word**: `.docx`, `.doc`

## 📝 File Naming Best Practices

Use descriptive names that include:
1. Document type
2. Insurance type 
3. Year/date
4. Unique identifier

### Examples:
```
data/raw/policies/
├── policy_auto_comprehensive_2024_001.pdf
├── policy_health_ppo_2024_002.pdf
├── policy_life_term_2024_003.pdf
└── policy_homeowners_2024_004.pdf

data/raw/claims/
├── claim_auto_collision_2024_101.pdf
├── claim_health_emergency_2024_102.pdf
├── claim_property_water_2024_103.pdf
└── claim_workers_comp_2024_104.pdf

data/raw/contracts/
├── contract_reinsurance_2024.pdf
├── contract_agent_agreement_2024.pdf
└── contract_service_provider_2024.pdf

data/raw/compliance/
├── compliance_hipaa_requirements_2024.pdf
├── compliance_state_regulations_ny_2024.pdf
└── compliance_privacy_policy_2024.pdf
```

## 🔒 Privacy & Security

### Automatic PII Removal
The system automatically detects and removes:
- Social Security Numbers (SSN)
- Phone numbers
- Email addresses
- Credit card numbers
- Account numbers
- Addresses
- Dates of birth

### Manual Review Recommended
- Review processed documents in `data/processed/`
- Verify PII removal is complete
- Check for any sensitive information that may need additional handling

## 📊 Data Quality Guidelines

### Document Quality
- Use clear, readable documents
- Avoid heavily redacted or corrupted files
- Ensure text is extractable (not just images)

### Coverage Balance
Try to include diverse examples:
- Different insurance types (auto, health, life, property)
- Various document types (policies, claims, contracts)
- Different complexity levels (simple to complex)
- Geographic diversity (different states/regulations)

### Minimum Data Requirements
- At least 10-20 documents per category for effective training
- Mix of document types within each category
- Representative examples of your use cases

## 🚀 Processing Workflow

1. **Place documents** in appropriate `data/raw/` subdirectories
2. **Run preprocessing** notebook (`01_data_preprocessing.ipynb`)
3. **Review processed data** in `data/processed/`
4. **Proceed with training** using remaining notebooks

## ❓ Common Questions

**Q: Can I use scanned documents?**
A: Yes, but ensure text is extractable. OCR quality affects training performance.

**Q: What if my documents contain sensitive information?**
A: The system removes common PII automatically, but review processed files to ensure compliance with your organization's data policies.

**Q: How many documents do I need?**
A: Minimum 50-100 documents across all categories, more is better for training quality.

**Q: Can I add more documents later?**
A: Yes, add new documents and re-run the preprocessing notebook.

## 🎯 Next Steps

After organizing your documents:
1. Run `notebooks/01_data_preprocessing.ipynb` to process your documents
2. Check the processed data quality
3. Proceed with tokenization and training

---

Need help? Check the main README.md or review the sample data templates in this directory.