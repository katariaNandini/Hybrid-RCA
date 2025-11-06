"""
Comprehensive evaluation script for RCAgent LLM model
"""

import json
import time
import random
from typing import Dict, List, Any, Tuple
from evaluation_metrics import RCAgentEvaluator, EvaluationResult
from rcagent_llm import RCAgentLLM

class RCAgentEvaluationSuite:
    """Comprehensive evaluation suite for RCAgent"""
    
    def __init__(self, model_name: str = "llama3"):
        self.rcagent = RCAgentLLM(model=model_name)
        self.evaluator = RCAgentEvaluator()
        self.results = []
        
    def evaluate_single_case(self, 
                           issue: str, 
                           files: List[str], 
                           ground_truth: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Evaluate a single test case
        
        Args:
            issue: Issue description
            files: List of file paths
            ground_truth: Optional ground truth for comparison
            
        Returns:
            Evaluation results
        """
        
        print(f"Evaluating case: {issue[:50]}...")
        
        # Run RCAgent analysis
        start_time = time.time()
        try:
            result = self.rcagent.analyze(issue, files)
            analysis_time = time.time() - start_time
            
            # Parse the result
            if isinstance(result, str):
                try:
                    result_json = json.loads(result)
                except json.JSONDecodeError:
                    result_json = {"root_cause": result, "recommendation": "Parse error"}
            else:
                result_json = result
                
        except Exception as e:
            print(f"Error in analysis: {e}")
            result_json = {"root_cause": f"Error: {e}", "recommendation": "Analysis failed"}
            analysis_time = 0
        
        # Extract telemetry data for evaluation
        telemetry_data = result_json.get("summary", {})
        
        # Evaluate if ground truth is provided
        evaluation_metrics = {}
        if ground_truth:
            # Root cause evaluation
            root_cause_metrics = self.evaluator.evaluate_root_cause_accuracy(
                result_json.get("root_cause", ""),
                ground_truth.get("root_cause", ""),
                telemetry_data
            )
            
            # Recommendation evaluation
            recommendation_metrics = self.evaluator.evaluate_recommendation_quality(
                result_json.get("recommendation", ""),
                ground_truth.get("recommendation", ""),
                telemetry_data
            )
            
            evaluation_metrics = {
                "root_cause_metrics": root_cause_metrics,
                "recommendation_metrics": recommendation_metrics
            }
        
        # Calculate confidence metrics
        confidence_metrics = self.evaluator.calculate_confidence_metrics(
            json.dumps(result_json), 
            telemetry_data
        )
        
        # Compile results
        case_result = {
            "issue": issue,
            "files": files,
            "analysis_time": analysis_time,
            "result": result_json,
            "telemetry_data": telemetry_data,
            "evaluation_metrics": evaluation_metrics,
            "confidence_metrics": confidence_metrics,
            "timestamp": time.time()
        }
        
        self.results.append(case_result)
        return case_result
    
    def evaluate_consistency(self, 
                           issue: str, 
                           files: List[str], 
                           num_runs: int = 3) -> Dict[str, float]:
        """
        Evaluate consistency across multiple runs
        """
        print(f"Evaluating consistency for: {issue[:50]}...")
        
        responses = []
        for i in range(num_runs):
            try:
                result = self.rcagent.analyze(issue, files)
                if isinstance(result, str):
                    result_json = json.loads(result)
                else:
                    result_json = result
                responses.append(json.dumps(result_json))
            except Exception as e:
                print(f"Error in run {i+1}: {e}")
                responses.append(f"Error: {e}")
        
        # Calculate consistency
        consistency_score = self.evaluator.evaluate_consistency(responses)
        
        return {
            "consistency_score": consistency_score,
            "num_runs": num_runs,
            "responses": responses
        }
    
    def evaluate_robustness(self, 
                           base_issue: str, 
                           base_files: List[str],
                           perturbations: List[Tuple[str, List[str]]]) -> Dict[str, float]:
        """
        Evaluate robustness to input perturbations
        """
        print(f"Evaluating robustness for: {base_issue[:50]}...")
        
        # Get base response
        try:
            base_result = self.rcagent.analyze(base_issue, base_files)
            if isinstance(base_result, str):
                base_result_json = json.loads(base_result)
            else:
                base_result_json = base_result
            base_response = json.dumps(base_result_json)
        except Exception as e:
            print(f"Error in base analysis: {e}")
            return {"robustness_score": 0.0, "error": str(e)}
        
        # Get perturbed responses
        perturbed_responses = []
        for perturbed_issue, perturbed_files in perturbations:
            try:
                result = self.rcagent.analyze(perturbed_issue, perturbed_files)
                if isinstance(result, str):
                    result_json = json.loads(result)
                else:
                    result_json = result
                perturbed_responses.append(json.dumps(result_json))
            except Exception as e:
                print(f"Error in perturbed analysis: {e}")
                perturbed_responses.append(f"Error: {e}")
        
        # Calculate robustness
        robustness_score = self.evaluator.evaluate_robustness(base_response, perturbed_responses)
        
        return {
            "robustness_score": robustness_score,
            "base_response": base_response,
            "perturbed_responses": perturbed_responses
        }
    
    def run_comprehensive_evaluation(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run comprehensive evaluation on multiple test cases
        """
        print("Starting comprehensive evaluation...")
        
        evaluation_results = {
            "test_cases": [],
            "overall_metrics": {},
            "consistency_results": [],
            "robustness_results": []
        }
        
        # Evaluate each test case
        for i, test_case in enumerate(test_cases):
            print(f"\n--- Test Case {i+1}/{len(test_cases)} ---")
            
            # Basic evaluation
            case_result = self.evaluate_single_case(
                test_case["issue"],
                test_case["files"],
                test_case.get("ground_truth")
            )
            evaluation_results["test_cases"].append(case_result)
            
            # Consistency evaluation (if requested)
            if test_case.get("evaluate_consistency", False):
                consistency_result = self.evaluate_consistency(
                    test_case["issue"],
                    test_case["files"],
                    test_case.get("consistency_runs", 3)
                )
                evaluation_results["consistency_results"].append(consistency_result)
            
            # Robustness evaluation (if requested)
            if test_case.get("evaluate_robustness", False):
                robustness_result = self.evaluate_robustness(
                    test_case["issue"],
                    test_case["files"],
                    test_case.get("perturbations", [])
                )
                evaluation_results["robustness_results"].append(robustness_result)
        
        # Calculate overall metrics
        evaluation_results["overall_metrics"] = self._calculate_overall_metrics()
        
        return evaluation_results
    
    def _calculate_overall_metrics(self) -> Dict[str, float]:
        """Calculate overall evaluation metrics"""
        if not self.results:
            return {}
        
        # Aggregate confidence metrics
        confidence_scores = [r["confidence_metrics"]["overall_confidence"] for r in self.results]
        avg_confidence = sum(confidence_scores) / len(confidence_scores)
        
        # Aggregate evaluation metrics (if available)
        root_cause_scores = []
        recommendation_scores = []
        
        for result in self.results:
            if "evaluation_metrics" in result and result["evaluation_metrics"]:
                if "root_cause_metrics" in result["evaluation_metrics"]:
                    root_cause_scores.append(
                        result["evaluation_metrics"]["root_cause_metrics"]["overall_accuracy"]
                    )
                if "recommendation_metrics" in result["evaluation_metrics"]:
                    recommendation_scores.append(
                        result["evaluation_metrics"]["recommendation_metrics"]["overall_quality"]
                    )
        
        return {
            "average_confidence": avg_confidence,
            "average_root_cause_accuracy": sum(root_cause_scores) / len(root_cause_scores) if root_cause_scores else 0.0,
            "average_recommendation_quality": sum(recommendation_scores) / len(recommendation_scores) if recommendation_scores else 0.0,
            "total_test_cases": len(self.results)
        }
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate a comprehensive evaluation report"""
        
        report = []
        report.append("=" * 60)
        report.append("RCAgent LLM Model Evaluation Report")
        report.append("=" * 60)
        report.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall metrics
        overall_metrics = results.get("overall_metrics", {})
        report.append("OVERALL METRICS:")
        report.append(f"  Average Confidence: {overall_metrics.get('average_confidence', 0):.3f}")
        report.append(f"  Average Root Cause Accuracy: {overall_metrics.get('average_root_cause_accuracy', 0):.3f}")
        report.append(f"  Average Recommendation Quality: {overall_metrics.get('average_recommendation_quality', 0):.3f}")
        report.append(f"  Total Test Cases: {overall_metrics.get('total_test_cases', 0)}")
        report.append("")
        
        # Individual test case results
        report.append("INDIVIDUAL TEST CASE RESULTS:")
        for i, test_case in enumerate(results.get("test_cases", [])):
            report.append(f"\n--- Test Case {i+1} ---")
            report.append(f"Issue: {test_case['issue'][:100]}...")
            report.append(f"Analysis Time: {test_case['analysis_time']:.2f}s")
            
            confidence = test_case.get("confidence_metrics", {})
            report.append(f"Confidence: {confidence.get('overall_confidence', 0):.3f}")
            report.append(f"Data Coverage: {confidence.get('data_coverage_confidence', 0):.3f}")
            report.append(f"Technical Confidence: {confidence.get('technical_confidence', 0):.3f}")
            
            if "evaluation_metrics" in test_case and test_case["evaluation_metrics"]:
                root_cause = test_case["evaluation_metrics"].get("root_cause_metrics", {})
                recommendation = test_case["evaluation_metrics"].get("recommendation_metrics", {})
                report.append(f"Root Cause Accuracy: {root_cause.get('overall_accuracy', 0):.3f}")
                report.append(f"Recommendation Quality: {recommendation.get('overall_quality', 0):.3f}")
        
        # Consistency results
        if results.get("consistency_results"):
            report.append("\nCONSISTENCY RESULTS:")
            for i, consistency in enumerate(results["consistency_results"]):
                report.append(f"  Test Case {i+1}: {consistency.get('consistency_score', 0):.3f}")
        
        # Robustness results
        if results.get("robustness_results"):
            report.append("\nROBUSTNESS RESULTS:")
            for i, robustness in enumerate(results["robustness_results"]):
                report.append(f"  Test Case {i+1}: {robustness.get('robustness_score', 0):.3f}")
        
        return "\n".join(report)

def create_sample_test_cases() -> List[Dict[str, Any]]:
    """Create sample test cases for evaluation"""
    
    return [
        {
            "issue": "High error rate in bank transfers",
            "files": [
                "C:\\Users\\rishi\\Downloads\\Bank\\telemetry\\2021_03_05\\log\\log_service.csv",
                "C:\\Users\\rishi\\Downloads\\Bank\\telemetry\\2021_03_05\\metric\\metric_app.csv",
                "C:\\Users\\rishi\\Downloads\\Bank\\telemetry\\2021_03_05\\trace\\trace_span.csv"
            ],
            "ground_truth": {
                "root_cause": "High latency in trace_span.csv causing performance issues",
                "recommendation": "Optimize database queries and implement caching"
            },
            "evaluate_consistency": True,
            "consistency_runs": 3
        },
        {
            "issue": "Database connection timeouts",
            "files": [
                "C:\\Users\\rishi\\Downloads\\Bank\\telemetry\\2021_03_05\\metric\\metric_container.csv",
                "C:\\Users\\rishi\\Downloads\\Bank\\telemetry\\2021_03_05\\trace\\trace_span.csv"
            ],
            "ground_truth": {
                "root_cause": "Database connection pool exhaustion",
                "recommendation": "Increase connection pool size and add connection monitoring"
            },
            "evaluate_robustness": True,
            "perturbations": [
                ("Database connection issues", [
                    "C:\\Users\\rishi\\Downloads\\Bank\\telemetry\\2021_03_05\\metric\\metric_container.csv"
                ]),
                ("Connection timeout errors", [
                    "C:\\Users\\rishi\\Downloads\\Bank\\telemetry\\2021_03_05\\trace\\trace_span.csv"
                ])
            ]
        }
    ]

def main():
    """Main evaluation function"""
    
    print("RCAgent LLM Model Evaluation Suite")
    print("=" * 50)
    
    # Initialize evaluation suite
    evaluator = RCAgentEvaluationSuite(model_name="llama3")
    
    # Create test cases
    test_cases = create_sample_test_cases()
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_evaluation(test_cases)
    
    # Generate and save report
    report = evaluator.generate_evaluation_report(results)
    
    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    with open("evaluation_report.txt", "w") as f:
        f.write(report)
    
    print("\nEvaluation completed!")
    print(f"Results saved to: evaluation_results.json")
    print(f"Report saved to: evaluation_report.txt")
    
    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(report)

if __name__ == "__main__":
    main()
