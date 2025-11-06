# RCAgent LLM Model Evaluation Metrics Guide

## Overview
This guide explains the comprehensive evaluation metrics and confidence measures for assessing the accuracy and precision of the RCAgent LLM model.

## 🎯 Core Evaluation Metrics

### 1. **Root Cause Analysis Accuracy**
- **Keyword Accuracy**: Measures overlap between predicted and ground truth technical terms
- **Technical Accuracy**: Evaluates alignment with actual telemetry data
- **Specificity Score**: Measures how specific vs generic the analysis is
- **Overall Accuracy**: Weighted combination of all accuracy measures

### 2. **Recommendation Quality**
- **Actionability Score**: Measures presence of specific, actionable steps
- **Relevance Score**: Evaluates how well recommendations address the actual problem
- **Completeness Score**: Assesses coverage of all problem aspects
- **Overall Quality**: Combined score of all recommendation metrics

### 3. **Confidence Metrics**
- **Data Coverage Confidence**: How well the response uses available telemetry data
- **Technical Confidence**: Use of specific technical terms and metrics
- **Consistency Score**: Internal consistency of the response
- **Uncertainty Score**: Detection of uncertainty indicators in the response

## 🔍 Detailed Metric Explanations

### Root Cause Analysis Metrics

#### Keyword Accuracy (0.0 - 1.0)
```python
# Extracts technical keywords from responses
technical_terms = ['latency', 'error', 'timeout', 'memory', 'cpu', 'database', 
                  'bottleneck', 'performance', 'throughput', 'response time', 
                  'p99', 'p95', 'garbage collection', 'heap', 'thread', 
                  'connection', 'query', 'index']

# Calculates Jaccard similarity between predicted and ground truth keywords
overlap = predicted_keywords ∩ ground_truth_keywords
union = predicted_keywords ∪ ground_truth_keywords
keyword_accuracy = len(overlap) / len(union)
```

#### Technical Accuracy (0.0 - 1.0)
```python
# Evaluates alignment with telemetry data
if 'latency' in root_cause and telemetry_data['worst_latency_p99'] > 0:
    score += 0.3
if 'error' in root_cause and telemetry_data['error_events_total'] > 0:
    score += 0.3
if worst_file_mentioned_in_root_cause:
    score += 0.4
```

#### Specificity Score (0.0 - 1.0)
```python
# Measures specificity vs genericity
generic_terms = ['issue', 'problem', 'error', 'bug', 'failure']
specific_terms = ['latency', 'timeout', 'memory', 'cpu', 'database', 'p99', 'p95']

specificity = specific_count / (generic_count + specific_count)
```

### Recommendation Quality Metrics

#### Actionability Score (0.0 - 1.0)
```python
# Detects actionable verbs and technical actions
action_indicators = ['implement', 'optimize', 'increase', 'decrease', 'configure', 
                    'tune', 'add', 'remove', 'enable', 'disable', 'upgrade', 
                    'scale', 'monitor']

technical_actions = ['heap size', 'connection pool', 'timeout', 'cache', 'index', 'query']

actionability = (action_count + technical_count) / 5
```

#### Relevance Score (0.0 - 1.0)
```python
# Checks if recommendations address the main telemetry issues
if high_latency_detected and latency_terms_in_recommendation:
    score += 0.5
if errors_detected and error_handling_terms_in_recommendation:
    score += 0.5
```

#### Completeness Score (0.0 - 1.0)
```python
# Evaluates coverage of different problem aspects
aspects = ['monitoring', 'scaling', 'optimization', 'configuration', 'testing']
covered_aspects = sum(1 for aspect in aspects if aspect in recommendation)

completeness = min(covered_aspects / 3, 1.0) * 0.5
```

### Confidence Metrics

#### Data Coverage Confidence (0.0 - 1.0)
```python
# Measures how well the response uses available telemetry data
if 'p99' in response and telemetry_data['worst_latency_p99'] > 0:
    score += 0.3
if 'error' in response and telemetry_data['error_events_total'] > 0:
    score += 0.3
if specific_files_mentioned:
    score += 0.4
```

#### Technical Confidence (0.0 - 1.0)
```python
# Measures use of specific technical terms and metrics
technical_terms = ['p99', 'p95', 'latency', 'throughput', 'memory', 'cpu', 'database']
technical_count = sum(1 for term in technical_terms if term in response)

numbers = re.findall(r'\d+\.?\d*', response)
technical_confidence = (technical_count / 3) * 0.5 + (0.3 if numbers else 0) + (0.2 if specific_files_mentioned else 0)
```

#### Consistency Score (0.0 - 1.0)
```python
# Detects contradictory statements
contradictions = [('high', 'low'), ('increase', 'decrease'), ('enable', 'disable'),
                  ('fast', 'slow'), ('good', 'bad')]

contradiction_count = sum(1 for pos, neg in contradictions if pos in response and neg in response)
consistency = max(0, 1 - (contradiction_count * 0.2))
```

#### Uncertainty Score (0.0 - 1.0)
```python
# Detects uncertainty indicators
uncertainty_terms = ['might', 'could', 'possibly', 'perhaps', 'maybe', 'likely', 
                    'probably', 'unclear', 'unknown', 'uncertain', 'ambiguous']

uncertainty_count = sum(1 for term in uncertainty_terms if term in response)
uncertainty_score = min(uncertainty_count / 3, 1.0)
```

## 🚀 Advanced Evaluation Metrics

### 1. **Consistency Evaluation**
- **Multiple Runs**: Run the same input multiple times and measure response similarity
- **Concept Similarity**: Extract key concepts and calculate overlap
- **Consistency Score**: Average pairwise similarity across runs

### 2. **Robustness Evaluation**
- **Input Perturbations**: Test with slightly modified inputs
- **Response Stability**: Measure how much responses change with small input changes
- **Robustness Score**: Average similarity between base and perturbed responses

### 3. **Performance Metrics**
- **Analysis Time**: Time taken to complete analysis
- **Memory Usage**: Memory consumption during analysis
- **Throughput**: Number of analyses per minute

## 📊 Evaluation Framework Usage

### Quick Evaluation
```python
from quick_evaluation import quick_evaluate_rcagent

# Run quick evaluation on your Bank dataset
results = quick_evaluate_rcagent()
print(f"Overall Score: {results['overall_score']:.3f}")
```

### Comprehensive Evaluation
```python
from evaluate_rcagent import RCAgentEvaluationSuite

# Create test cases
test_cases = [
    {
        "issue": "High error rate in bank transfers",
        "files": ["path/to/telemetry/files"],
        "ground_truth": {
            "root_cause": "Expected root cause",
            "recommendation": "Expected recommendation"
        }
    }
]

# Run comprehensive evaluation
evaluator = RCAgentEvaluationSuite()
results = evaluator.run_comprehensive_evaluation(test_cases)
```

### Consistency Testing
```python
from quick_evaluation import run_consistency_test

# Test consistency across multiple runs
consistency_score = run_consistency_test()
print(f"Consistency: {consistency_score:.3f}")
```

## 🎯 Interpretation Guidelines

### Score Ranges
- **0.8 - 1.0**: Excellent - High quality, reliable analysis
- **0.6 - 0.8**: Good - Solid analysis with minor improvements needed
- **0.4 - 0.6**: Fair - Analysis needs significant improvement
- **0.0 - 0.4**: Poor - Analysis quality is low

### Key Indicators
- **High Confidence + Low Accuracy**: Model is overconfident
- **Low Confidence + High Accuracy**: Model is underconfident
- **Low Consistency**: Model responses are unstable
- **Low Robustness**: Model is sensitive to input changes

## 🔧 Improving Model Performance

### Based on Evaluation Results

1. **Low Accuracy**:
   - Improve prompt engineering
   - Add more specific instructions
   - Include better examples in prompts

2. **Low Confidence**:
   - Ensure telemetry data is complete
   - Add more specific metrics to prompts
   - Improve data preprocessing

3. **Low Consistency**:
   - Add temperature control to LLM
   - Implement response validation
   - Use more deterministic prompts

4. **Low Robustness**:
   - Test with more diverse inputs
   - Add input validation
   - Implement fallback mechanisms

## 📈 Continuous Monitoring

### Automated Evaluation
- Set up automated evaluation pipelines
- Monitor metrics over time
- Alert on performance degradation

### A/B Testing
- Compare different model versions
- Test prompt variations
- Evaluate new features

### Human Validation
- Expert review of high-impact cases
- Crowdsourced evaluation
- Domain expert feedback

## 🎯 Best Practices

1. **Regular Evaluation**: Run evaluations after any model changes
2. **Diverse Test Cases**: Include various types of issues and data
3. **Ground Truth**: Use expert-validated ground truth when possible
4. **Continuous Improvement**: Use evaluation results to improve the model
5. **Documentation**: Keep detailed records of evaluation results

This comprehensive evaluation framework ensures that your RCAgent LLM model provides accurate, reliable, and actionable root cause analysis for your Bank dataset and other telemetry data.
