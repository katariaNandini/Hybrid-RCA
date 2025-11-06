"""
Evaluation metrics and confidence measures for RCAgent LLM model
"""

import json
import re
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

@dataclass
class EvaluationResult:
    """Container for evaluation results"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confidence_score: float
    consistency_score: float
    completeness_score: float
    specificity_score: float

class RCAgentEvaluator:
    """Evaluation framework for RCAgent LLM model"""
    
    def __init__(self):
        self.ground_truth_labels = []
        self.predicted_labels = []
        self.confidence_scores = []
        
    def evaluate_root_cause_accuracy(self, 
                                   predicted_root_cause: str, 
                                   ground_truth_root_cause: str,
                                   telemetry_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate accuracy of root cause identification
        
        Metrics:
        1. Semantic similarity (using keyword matching)
        2. Technical accuracy (based on telemetry data)
        3. Specificity score
        """
        
        # 1. Keyword-based accuracy
        predicted_keywords = self._extract_keywords(predicted_root_cause)
        ground_truth_keywords = self._extract_keywords(ground_truth_root_cause)
        
        keyword_accuracy = self._calculate_keyword_overlap(predicted_keywords, ground_truth_keywords)
        
        # 2. Technical accuracy based on telemetry data
        technical_accuracy = self._evaluate_technical_accuracy(
            predicted_root_cause, telemetry_data
        )
        
        # 3. Specificity score (how specific vs generic)
        specificity = self._calculate_specificity_score(predicted_root_cause)
        
        return {
            "keyword_accuracy": keyword_accuracy,
            "technical_accuracy": technical_accuracy,
            "specificity_score": specificity,
            "overall_accuracy": (keyword_accuracy + technical_accuracy + specificity) / 3
        }
    
    def evaluate_recommendation_quality(self, 
                                      predicted_recommendation: str,
                                      ground_truth_recommendation: str,
                                      telemetry_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate quality of recommendations
        
        Metrics:
        1. Actionability score
        2. Technical relevance
        3. Completeness
        """
        
        # 1. Actionability score (presence of specific actions)
        actionability = self._calculate_actionability_score(predicted_recommendation)
        
        # 2. Technical relevance to the problem
        relevance = self._calculate_relevance_score(
            predicted_recommendation, telemetry_data
        )
        
        # 3. Completeness (covers all aspects of the problem)
        completeness = self._calculate_completeness_score(
            predicted_recommendation, telemetry_data
        )
        
        return {
            "actionability_score": actionability,
            "relevance_score": relevance,
            "completeness_score": completeness,
            "overall_quality": (actionability + relevance + completeness) / 3
        }
    
    def calculate_confidence_metrics(self, 
                                    llm_response: str,
                                    telemetry_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate confidence metrics for LLM predictions
        
        Metrics:
        1. Data coverage confidence
        2. Response consistency
        3. Technical confidence
        4. Uncertainty indicators
        """
        
        # 1. Data coverage confidence
        data_coverage = self._calculate_data_coverage_confidence(
            llm_response, telemetry_data
        )
        
        # 2. Response consistency (internal consistency)
        consistency = self._calculate_response_consistency(llm_response)
        
        # 3. Technical confidence (use of specific metrics)
        technical_confidence = self._calculate_technical_confidence(
            llm_response, telemetry_data
        )
        
        # 4. Uncertainty indicators
        uncertainty = self._detect_uncertainty_indicators(llm_response)
        
        return {
            "data_coverage_confidence": data_coverage,
            "consistency_score": consistency,
            "technical_confidence": technical_confidence,
            "uncertainty_score": uncertainty,
            "overall_confidence": (data_coverage + consistency + technical_confidence - uncertainty) / 4
        }
    
    def evaluate_consistency(self, 
                           multiple_responses: List[str]) -> float:
        """
        Evaluate consistency across multiple runs of the same input
        """
        if len(multiple_responses) < 2:
            return 1.0
            
        # Extract key concepts from each response
        concepts = [self._extract_key_concepts(response) for response in multiple_responses]
        
        # Calculate pairwise similarity
        similarities = []
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                similarity = self._calculate_concept_similarity(concepts[i], concepts[j])
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def evaluate_robustness(self, 
                          base_response: str,
                          perturbed_responses: List[str]) -> float:
        """
        Evaluate robustness to input perturbations
        """
        base_concepts = self._extract_key_concepts(base_response)
        
        similarities = []
        for perturbed_response in perturbed_responses:
            perturbed_concepts = self._extract_key_concepts(perturbed_response)
            similarity = self._calculate_concept_similarity(base_concepts, perturbed_concepts)
            similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    # Helper methods
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract technical keywords from text"""
        # Technical terms related to system issues
        technical_terms = [
            'latency', 'error', 'timeout', 'memory', 'cpu', 'database', 'network',
            'bottleneck', 'performance', 'throughput', 'response time', 'p99', 'p95',
            'garbage collection', 'heap', 'thread', 'connection', 'query', 'index'
        ]
        
        text_lower = text.lower()
        found_keywords = [term for term in technical_terms if term in text_lower]
        
        # Also extract numbers (metrics)
        numbers = re.findall(r'\d+\.?\d*', text)
        found_keywords.extend(numbers)
        
        return found_keywords
    
    def _calculate_keyword_overlap(self, predicted: List[str], ground_truth: List[str]) -> float:
        """Calculate overlap between predicted and ground truth keywords"""
        if not ground_truth:
            return 0.0
        
        predicted_set = set(predicted)
        ground_truth_set = set(ground_truth)
        
        intersection = predicted_set.intersection(ground_truth_set)
        union = predicted_set.union(ground_truth_set)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _evaluate_technical_accuracy(self, root_cause: str, telemetry_data: Dict[str, Any]) -> float:
        """Evaluate if root cause aligns with telemetry data"""
        score = 0.0
        
        # Check if root cause mentions specific metrics from telemetry
        if 'latency' in root_cause.lower() and telemetry_data.get('worst_latency_p99', 0) > 0:
            score += 0.3
        
        if 'error' in root_cause.lower() and telemetry_data.get('error_events_total', 0) > 0:
            score += 0.3
        
        # Check if root cause mentions specific files
        worst_file = telemetry_data.get('worst_latency_file', '')
        if worst_file and any(part in root_cause.lower() for part in worst_file.split('\\')[-1].split('_')):
            score += 0.4
        
        return min(score, 1.0)
    
    def _calculate_specificity_score(self, text: str) -> float:
        """Calculate how specific vs generic the response is"""
        generic_terms = ['issue', 'problem', 'error', 'bug', 'failure']
        specific_terms = ['latency', 'timeout', 'memory', 'cpu', 'database', 'p99', 'p95']
        
        text_lower = text.lower()
        generic_count = sum(1 for term in generic_terms if term in text_lower)
        specific_count = sum(1 for term in specific_terms if term in text_lower)
        
        if generic_count + specific_count == 0:
            return 0.5
        
        return specific_count / (generic_count + specific_count)
    
    def _calculate_actionability_score(self, recommendation: str) -> float:
        """Calculate how actionable the recommendation is"""
        action_indicators = [
            'implement', 'optimize', 'increase', 'decrease', 'configure', 'tune',
            'add', 'remove', 'enable', 'disable', 'upgrade', 'scale', 'monitor'
        ]
        
        text_lower = recommendation.lower()
        action_count = sum(1 for action in action_indicators if action in text_lower)
        
        # Also check for specific technical actions
        technical_actions = [
            'heap size', 'connection pool', 'timeout', 'cache', 'index', 'query'
        ]
        technical_count = sum(1 for action in technical_actions if action in text_lower)
        
        return min((action_count + technical_count) / 5, 1.0)
    
    def _calculate_relevance_score(self, recommendation: str, telemetry_data: Dict[str, Any]) -> float:
        """Calculate relevance of recommendation to the problem"""
        score = 0.0
        
        # Check if recommendation addresses the main issue
        if telemetry_data.get('worst_latency_p99', 0) > 1000:  # High latency
            if any(term in recommendation.lower() for term in ['latency', 'performance', 'optimize', 'cache']):
                score += 0.5
        
        if telemetry_data.get('error_events_total', 0) > 0:  # Has errors
            if any(term in recommendation.lower() for term in ['error', 'exception', 'retry', 'fallback']):
                score += 0.5
        
        return min(score, 1.0)
    
    def _calculate_completeness_score(self, recommendation: str, telemetry_data: Dict[str, Any]) -> float:
        """Calculate completeness of recommendation"""
        score = 0.0
        
        # Check if recommendation covers multiple aspects
        aspects = ['monitoring', 'scaling', 'optimization', 'configuration', 'testing']
        covered_aspects = sum(1 for aspect in aspects if aspect in recommendation.lower())
        
        score += min(covered_aspects / 3, 1.0) * 0.5
        
        # Check if recommendation is specific to the data
        if telemetry_data.get('files', 0) > 1:
            if 'multiple' in recommendation.lower() or 'all' in recommendation.lower():
                score += 0.5
        
        return min(score, 1.0)
    
    def _calculate_data_coverage_confidence(self, response: str, telemetry_data: Dict[str, Any]) -> float:
        """Calculate confidence based on data coverage"""
        score = 0.0
        
        # Check if response uses specific metrics from telemetry
        if 'p99' in response.lower() and telemetry_data.get('worst_latency_p99', 0) > 0:
            score += 0.3
        
        if 'error' in response.lower() and telemetry_data.get('error_events_total', 0) > 0:
            score += 0.3
        
        # Check if response mentions specific files
        files_mentioned = sum(1 for file_info in telemetry_data.get('files_summaries', [])
                            if any(part in response.lower() for part in file_info.get('path', '').split('\\')[-1].split('_')))
        if files_mentioned > 0:
            score += 0.4
        
        return min(score, 1.0)
    
    def _calculate_response_consistency(self, response: str) -> float:
        """Calculate internal consistency of response"""
        # Check for contradictory statements
        contradictions = [
            ('high', 'low'), ('increase', 'decrease'), ('enable', 'disable'),
            ('fast', 'slow'), ('good', 'bad')
        ]
        
        text_lower = response.lower()
        contradiction_count = 0
        
        for pos, neg in contradictions:
            if pos in text_lower and neg in text_lower:
                contradiction_count += 1
        
        return max(0, 1 - (contradiction_count * 0.2))
    
    def _calculate_technical_confidence(self, response: str, telemetry_data: Dict[str, Any]) -> float:
        """Calculate confidence based on technical specificity"""
        score = 0.0
        
        # Check for specific technical terms
        technical_terms = ['p99', 'p95', 'latency', 'throughput', 'memory', 'cpu', 'database']
        technical_count = sum(1 for term in technical_terms if term in response.lower())
        score += min(technical_count / 3, 1.0) * 0.5
        
        # Check for specific numbers/metrics
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            score += 0.3
        
        # Check for specific file references
        if any(file_info.get('path', '') in response for file_info in telemetry_data.get('files_summaries', [])):
            score += 0.2
        
        return min(score, 1.0)
    
    def _detect_uncertainty_indicators(self, response: str) -> float:
        """Detect uncertainty indicators in response"""
        uncertainty_terms = [
            'might', 'could', 'possibly', 'perhaps', 'maybe', 'likely', 'probably',
            'unclear', 'unknown', 'uncertain', 'unclear', 'ambiguous'
        ]
        
        text_lower = response.lower()
        uncertainty_count = sum(1 for term in uncertainty_terms if term in text_lower)
        
        return min(uncertainty_count / 3, 1.0)
    
    def _extract_key_concepts(self, response: str) -> List[str]:
        """Extract key concepts from response"""
        # Extract technical terms, numbers, and key phrases
        concepts = []
        
        # Technical terms
        technical_terms = re.findall(r'\b(?:latency|error|timeout|memory|cpu|database|performance|bottleneck)\b', response.lower())
        concepts.extend(technical_terms)
        
        # Numbers
        numbers = re.findall(r'\d+\.?\d*', response)
        concepts.extend(numbers)
        
        # Key phrases (simplified)
        phrases = re.findall(r'\b\w+\s+\w+\b', response.lower())
        concepts.extend(phrases[:5])  # Limit to first 5 phrases
        
        return concepts
    
    def _calculate_concept_similarity(self, concepts1: List[str], concepts2: List[str]) -> float:
        """Calculate similarity between concept lists"""
        if not concepts1 or not concepts2:
            return 0.0
        
        set1 = set(concepts1)
        set2 = set(concepts2)
        
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        
        return len(intersection) / len(union) if union else 0.0

def run_evaluation_example():
    """Example of how to use the evaluation framework"""
    
    # Example telemetry data
    telemetry_data = {
        "files": 4,
        "rows_total": 7143883,
        "error_events_total": 0,
        "worst_latency_p99": 3167.0,
        "worst_latency_file": "trace_span.csv",
        "files_summaries": [
            {
                "path": "log_service.csv",
                "rows": 450361,
                "indicators": {"error_events": 0, "latency": {}}
            }
        ]
    }
    
    # Example LLM response
    llm_response = {
        "root_cause": "High latency in trace_span.csv with p99 of 3167ms indicates database bottleneck",
        "recommendation": "Optimize database queries and implement caching to reduce latency"
    }
    
    # Ground truth (what we expect)
    ground_truth = {
        "root_cause": "Database performance issues causing high latency",
        "recommendation": "Optimize database queries and add caching"
    }
    
    # Initialize evaluator
    evaluator = RCAgentEvaluator()
    
    # Evaluate root cause accuracy
    root_cause_metrics = evaluator.evaluate_root_cause_accuracy(
        llm_response["root_cause"], 
        ground_truth["root_cause"], 
        telemetry_data
    )
    
    # Evaluate recommendation quality
    recommendation_metrics = evaluator.evaluate_recommendation_quality(
        llm_response["recommendation"], 
        ground_truth["recommendation"], 
        telemetry_data
    )
    
    # Calculate confidence metrics
    confidence_metrics = evaluator.calculate_confidence_metrics(
        json.dumps(llm_response), 
        telemetry_data
    )
    
    print("=== RCAgent Evaluation Results ===")
    print(f"Root Cause Accuracy: {root_cause_metrics['overall_accuracy']:.3f}")
    print(f"Recommendation Quality: {recommendation_metrics['overall_quality']:.3f}")
    print(f"Overall Confidence: {confidence_metrics['overall_confidence']:.3f}")
    
    return {
        "root_cause_metrics": root_cause_metrics,
        "recommendation_metrics": recommendation_metrics,
        "confidence_metrics": confidence_metrics
    }

if __name__ == "__main__":
    run_evaluation_example()
