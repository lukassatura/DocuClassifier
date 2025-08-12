# DocuClassifier
Document classification system using Amazon Bedrock's runtime.

## Overview
Production-ready document classification system using Amazon Bedrock's LLM runtime to automatically classify text documents into discovered categories. Features dynamic JSON schema generation, structured output validation, and adaptive prompt engineering.

## 🏗️ Core Architecture

```python
# Main Components
DataLoader          # File parsing and document extraction
BedrockClassifier   # LLM classification engine with tool use
SchemaGenerator     # Dynamic JSON schema creation
PromptGenerator     # Context-aware prompt construction
```

### Key Innovation: Dynamic Structured Output Enforcement

**Problem**: LLMs produce inconsistent, unstructured responses  
**Solution**: Runtime JSON schema generation with Bedrock tool use

```python
# Schema adapts to discovered categories automatically
def generate_classification_schema(categories: set[int], include_reasoning: bool = True):
    return {
        "type": "object",
        "properties": {
            "category": {"type": "integer", "enum": sorted(list(categories))},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning": {"type": "string", "minLength": 10}
        },
        "required": ["category", "confidence"],
        "additionalProperties": False
    }
```

**Benefits**:
- Enforces valid category predictions only
- Guarantees response structure consistency  
- Eliminates parsing errors from malformed JSON
- Provides confidence scoring and reasoning

## 🎯 Technical Features

### 1. Adaptive Category Discovery
```python
def discover_categories(self, documents) -> set[int]:
    """Automatically detects categories from training data"""
    categories = {doc.category for doc in documents}
    # Updates schemas, prompts, and validation rules dynamically
```

### 2. Intelligent Few-Shot Learning
```python
def prepare_training_examples(self, documents, samples_per_category=3):
    """Selects representative examples for each category"""
    # Optimizes context window usage
    # Ensures balanced category representation
    # Improves classification accuracy through contextual examples
```

### 3. Chain-of-Thought Reasoning
```python
# Structured reasoning process
prompt = """Analyze this document step by step:
1. Identify main topics and themes
2. Match to most appropriate category
3. Assess confidence level (0.0-1.0)
4. Provide reasoning for decision"""
```

### 4. Comprehensive Validation Pipeline
```python
class ClassificationResult(BaseModel):
    category: int = Field(..., ge=1, description="Predicted category")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    reasoning: Optional[str] = Field(None, description="Classification reasoning")
    
    @field_validator('category')
    @classmethod
    def validate_category_range(cls, v):
        if v not in cls._valid_categories:
            raise ValueError(f"Category {v} not in valid categories")
        return v
```

## 🔧 Implementation Details

### Bedrock Integration with Tool Use
```python
def _invoke_model_with_tool(self, prompt, tool_name, tool_description, schema):
    """Converse API with structured output enforcement"""
    tool_config = {
        "tools": [{
            "toolSpec": {
                "name": tool_name,
                "description": tool_description,
                "inputSchema": {"json": schema}
            }
        }],
        "toolChoice": {"tool": {"name": tool_name}}
    }
    # Forces model to use structured output format
```

### Error Handling & Resilience
- **Exponential Backoff**: 3 retries with increasing delays
- **Graceful Degradation**: Fallback to default categories on failure
- **Input Validation**: Pydantic models for all data structures
- **Response Parsing**: Multiple extraction methods for robustness

### Performance Optimizations
- **Rate Limiting**: 0.5s delays prevent API throttling
- **Batch Processing**: Efficient multi-document handling
- **Memory Management**: Streaming for large datasets
- **Caching**: Persistent category descriptions

## 📈 Evaluation Framework

### Metrics Collection
```python
# Comprehensive performance analysis
- Accuracy: Overall classification success rate
- F1 Scores: Macro, weighted, per-class performance
- Confusion Matrix: Misclassification pattern analysis  
- Confidence Distribution: Prediction reliability assessment
```

### Quality Assurance
- **Type Safety**: Complete type annotations
- **Schema Validation**: Runtime response verification
- **Error Pattern Analysis**: Systematic misclassification identification
- **Confidence Thresholding**: Low-confidence prediction flagging

## 🚀 Advanced Capabilities

### Dynamic Prompt Construction
```python
def generate_classification_prompt(text, category_descriptions, training_examples, categories):
    """Context-aware prompt building"""
    # Includes relevant category descriptions
    # Adds few-shot examples for each category  
    # Adapts to available training data
    # Optimizes prompt length vs. context richness
```

### Persistent State Management
```python
# Automatic save/load of generated category descriptions
{
    "descriptions": {category_id: description},
    "metadata": {
        "categories": [discovered_categories],
        "num_categories": auto_detected,
        "generated_at": "timestamp"
    }
}
```

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.8+** 
- **AWS Account** with Bedrock access
- **Valid AWS Bedrock API Token** with permissions for:
  - `bedrock:InvokeModel`
  - `bedrock:ListFoundationModels`
  - Access to `anthropic.claude-sonnet-4` model

### Step 1: Environment Setup
```bash
# Clone the repository
git clone <repository-url>
cd DocuClassifier

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: AWS Configuration
```bash
# Set your Bedrock API token
export AWS_BEARER_TOKEN_BEDROCK="your_bedrock_api_token_here"

# Alternatively, set it directly in the notebook:
# os.environ['AWS_BEARER_TOKEN_BEDROCK'] = "your_token_here"
```

### Step 3: Data Preparation
Ensure your training and testing data files are in the project directory:
- `trainingdata.txt`: Training documents with category labels
- `testingdata.txt`: Test documents for evaluation

**Expected format:**
```
<number_of_documents>
<category_id> <document_text>
<category_id> <document_text>
...
```

### Step 4: Run Classification
```bash
# Start Jupyter notebook
jupyter notebook

# Open and run docuclassifier.ipynb
# Or run directly with Python:
python -c "exec(open('docuclassifier.py').read())"
```

## 📊 Usage & Configuration

### Quick Start
```python
# Initialize and run classification
classifier = BedrockClassifier(runtime_client)
classifier.prepare_training_examples(train_documents, samples_per_category=3)
predictions = classifier.classify_batch(test_documents, use_cot=True)
```

### Key Parameters
- **Model**: `us.anthropic.claude-sonnet-4-20250514-v1:0` (can be overridden)
- **Temperature**: `0.1` (consistent outputs)
- **Max Tokens**: `4000` (detailed reasoning)
- **Examples per Category**: `3` (optimal context/performance balance)

### Output Files
- `classification_results.csv`: Predictions with confidence and reasoning
- `category_descriptions.json`: LLM-generated category definitions

## 🎛️ Technical Highlights

### Innovation Areas
1. **Runtime Schema Generation**: JSON schemas created based on discovered data patterns
2. **Adaptive Validation**: Validation rules adjust to actual category distributions  
3. **Structured LLM Interaction**: Tool use enforcement prevents unstructured responses
4. **Context-Aware Processing**: Prompts and examples adapt to available training data

### Production Features
- **Comprehensive Logging**: Structured logs for monitoring and debugging
- **Error Recovery**: Multiple fallback mechanisms for robust operation
- **Performance Monitoring**: Real-time metrics collection and analysis
- **Scalable Architecture**: Designed for enterprise workloads

## 🔍 Troubleshooting

### Common Issues
- **Authentication Error**: Verify AWS_BEARER_TOKEN_BEDROCK is set correctly
- **Model Access**: Ensure your AWS account has access to Claude Sonnet models
- **Rate Limiting**: Increase delay between requests if hitting API limits
- **Memory Issues**: Reduce batch size for large datasets

### Dependencies
```bash
# requirements.txt
boto3>=1.34.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
numpy>=2.3.0
seaborn>=0.12.0
tqdm>=4.65.0
pydantic>=2.0.0
```

---

**Technical Implementation**: This solution demonstrates advanced LLM integration patterns including dynamic schema generation, structured output validation, and adaptive prompt engineering for scalable document classification systems.
