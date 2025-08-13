# DSPy Implementation for Document Classification

## Overview

This repository contains an enhanced document classification system using **DSPy (Declarative Self-improving Python)** framework with Amazon Bedrock. The DSPy implementation (`assignment_dspy.ipynb`) provides a more robust and systematically optimized approach compared to manual prompt engineering.

## What is DSPy?

DSPy is a framework for algorithmically optimizing LM prompts and weights. Instead of manually crafting prompts, DSPy:
- Automatically optimizes prompts based on training data
- Selects the best few-shot examples
- Provides systematic optimization metrics
- Makes the system more maintainable and reproducible

## Key Improvements Over Manual Prompting

### 1. **Automatic Prompt Optimization**
- **Original**: Manual crafting of prompts with trial and error
- **DSPy**: Systematic optimization using `BootstrapFewShot` that automatically finds the best prompt structure

### 2. **Robustness**
- **Original**: Sensitive to exact prompt wording
- **DSPy**: Less sensitive to variations, more consistent results

### 3. **Code Structure**
- **Original**: String manipulation and template filling
- **DSPy**: Declarative signatures and modules with clean separation of concerns

### 4. **Few-Shot Learning**
- **Original**: Manual selection of examples
- **DSPy**: Automatic selection of optimal examples from training data

### 5. **Chain-of-Thought Reasoning**
- **Original**: Manually implemented in prompts
- **DSPy**: Built-in `ChainOfThought` module

## Implementation Components

### DSPy Signatures
```python
class ClassifyDocument(dspy.Signature):
    """Classify a financial/business document into one of 8 categories."""
    document = dspy.InputField(desc="The document text to classify")
    category_descriptions = dspy.InputField(desc="Descriptions of each category (1-8)")
    category = dspy.OutputField(desc="Category number (1-8)")
    confidence = dspy.OutputField(desc="Confidence score (0.0-1.0)")
    reasoning = dspy.OutputField(desc="Brief explanation of the classification")
```

### DSPy Modules
```python
class DocumentClassifierDSPy(dspy.Module):
    """DSPy-based document classifier with Chain of Thought reasoning"""
    def __init__(self):
        self.classify = dspy.ChainOfThought(ClassifyDocument)
```

### Optimization Process
```python
optimizer = BootstrapFewShot(
    metric=classification_metric,
    max_bootstrapped_demos=3,
    max_labeled_demos=5,
    max_rounds=2
)
optimized_classifier = optimizer.compile(classifier_dspy, trainset, valset)
```

## Benefits Demonstrated

### 1. **Performance Metrics**
- Maintains high accuracy comparable to manual prompting
- Provides consistent F1 scores across categories
- Better confidence calibration

### 2. **Development Efficiency**
- Reduced time spent on prompt engineering
- Automatic hyperparameter tuning
- Easier to iterate and improve

### 3. **Maintainability**
- Cleaner codebase
- Easier to modify for new requirements
- Better documentation through declarative signatures

### 4. **Reproducibility**
- Systematic optimization process
- Consistent results across runs
- Clear optimization metrics

## Usage

### Prerequisites
```bash
pip install dspy-ai
pip install boto3 pandas scikit-learn
```

### Running the DSPy Implementation
1. Open `assignment_dspy.ipynb` in Jupyter
2. Ensure AWS credentials are configured
3. Run all cells to:
   - Load and prepare data
   - Configure DSPy with Bedrock
   - Optimize the classifier
   - Evaluate on test data
   - Compare with manual approach

### Key Files
- `assignment_dspy.ipynb`: DSPy implementation notebook
- `assignment.ipynb`: Original manual prompting implementation
- `classification_results_dspy.csv`: DSPy classification results
- `category_descriptions_dspy.json`: Generated category descriptions
- `dspy_optimization_info.txt`: Optimization summary

## DSPy Architecture

```
┌─────────────────┐
│   Training Data │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  DSPy Examples  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ BootstrapFewShot│
│   Optimizer     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Optimized     │
│   Classifier    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Test Results   │
└─────────────────┘
```

## Comparison Results

| Metric | Manual Prompting | DSPy Implementation |
|--------|-----------------|---------------------|
| Code Complexity | High (manual prompt crafting) | Low (declarative) |
| Optimization | Manual trial & error | Automatic |
| Robustness | Sensitive to wording | More stable |
| Maintainability | Difficult | Easy |
| Reproducibility | Variable | Consistent |

## Future Enhancements

1. **Advanced Optimizers**
   - Experiment with `MIPRO` for better optimization
   - Try `BootstrapFewShotWithRandomSearch`

2. **Ensemble Methods**
   - Combine multiple DSPy classifiers
   - Implement voting mechanisms

3. **Retrieval Augmentation**
   - Add DSPy retrieval modules
   - Enhance context with similar examples

4. **Hyperparameter Tuning**
   - Optimize temperature settings
   - Adjust demonstration counts

5. **Cross-validation**
   - Implement k-fold validation
   - Better generalization testing

## Key Takeaways

1. **DSPy eliminates manual prompt engineering** - The framework automatically finds optimal prompts through systematic optimization.

2. **Improved robustness** - Less sensitive to exact wording, more consistent across different inputs.

3. **Better maintainability** - Declarative approach makes the code easier to understand and modify.

4. **Systematic improvement** - Clear metrics and optimization process for continuous enhancement.

5. **Production-ready** - The DSPy implementation is more suitable for production environments due to its robustness and maintainability.

## Conclusion

The DSPy implementation demonstrates significant advantages over manual prompting:
- **Automatic optimization** replaces tedious manual tuning
- **Cleaner code** through declarative programming
- **Better robustness** through systematic optimization
- **Easier maintenance** and future improvements

This makes DSPy the preferred approach for building robust, production-ready LLM applications that need to be maintained and improved over time.

## References

- [DSPy Documentation](https://github.com/stanfordnlp/dspy)
- [DSPy Paper](https://arxiv.org/abs/2310.03714)
- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
