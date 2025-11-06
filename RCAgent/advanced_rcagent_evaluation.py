"""
Advanced RCAgent evaluation using semantic metrics and GPT-4 scoring
"""

import json
import time
from typing import Dict, List, Any
from advanced_evaluation_metrics import AdvancedRCAgentEvaluator, AdvancedEvaluationResult
from rcagent_llm import RCAgentLLM

class AdvancedRCAgentEvaluationSuite:
    """Advanced evaluation suite using semantic metrics and GPT-4"""
    
    def __init__(self, model_name: str = "llama3", openai_api_key: str = None):
        self.rcagent = RCAgentLLM(model=model_name)
        self.evaluator = AdvancedRCAgentEvaluator(openai_api_key=openai_api_key)
        self.results = []
        
    def evaluate_with_advanced_metrics(self, 
                                      issue: str, 
                                      files: List[str], 
                                      ground_truth: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Evaluate using advanced semantic metrics and GPT-4 scoring
        """
        
        print(f"Advanced evaluation for: {issue[:50]}...")
        
        # Run RCAgent analysis
        start_time = time.time()
        try:
            result = self.rcagent.analyze(issue, files)
            analysis_time = time.time() - start_time
            
            # Parse result
            if isinstance(result, str):
                result_json = json.loads(result)
            else:
                result_json = result
                
        except Exception as e:
            print(f"Error in analysis: {e}")
            result_json = {"root_cause": f"Error: {e}", "recommendation": "Analysis failed"}
            analysis_time = 0
        
        # Extract telemetry data
        telemetry_data = result_json.get("summary", {})
        predicted_root_cause = result_json.get("root_cause", "")
        predicted_recommendation = result_json.get("recommendation", "")
        
        # Run advanced evaluation if ground truth is provided
        advanced_metrics = {}
        if ground_truth:
            try:
                # Run comprehensive evaluation
                evaluation_result = self.evaluator.comprehensive_evaluation(
                    predicted_root_cause, predicted_recommendation,
                    ground_truth.get("root_cause", ""), ground_truth.get("recommendation", ""),
                    telemetry_data
                )
                
                advanced_metrics = {
                    "meteor_score": evaluation_result.meteor_score,
                    "rouge_scores": evaluation_result.rouge_scores,
                    "bert_score": evaluation_result.bert_score,
                    "embedding_score": evaluation_result.embedding_score,
                    "gpt4_correctness": evaluation_result.gpt4_correctness,
                    "gpt4_helpfulness": evaluation_result.gpt4_helpfulness,
                    "overall_score": evaluation_result.overall_score
                }
                
            except Exception as e:
                print(f"Advanced evaluation error: {e}")
                advanced_metrics = {"error": str(e)}
        
        # Compile results
        case_result = {
            "issue": issue,
            "files": files,
            "analysis_time": analysis_time,
            "result": result_json,
            "telemetry_data": telemetry_data,
            "advanced_metrics": advanced_metrics,
            "timestamp": time.time()
        }
        
        self.results.append(case_result)
        return case_result
    
    def run_comprehensive_advanced_evaluation(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run comprehensive evaluation using advanced metrics
        """
        print("Starting advanced comprehensive evaluation...")
        
        evaluation_results = {
            "test_cases": [],
            "overall_metrics": {},
            "advanced_metrics_summary": {}
        }
        
        # Evaluate each test case
        for i, test_case in enumerate(test_cases):
            print(f"\n--- Advanced Test Case {i+1}/{len(test_cases)} ---")
            
            case_result = self.evaluate_with_advanced_metrics(
                test_case["issue"],
                test_case["files"],
                test_case.get("ground_truth")
            )
            evaluation_results["test_cases"].append(case_result)
        
        # Calculate overall metrics
        evaluation_results["overall_metrics"] = self._calculate_overall_advanced_metrics()
        evaluation_results["advanced_metrics_summary"] = self._calculate_advanced_metrics_summary()
        
        return evaluation_results
    
    def _calculate_overall_advanced_metrics(self) -> Dict[str, float]:
        """Calculate overall advanced metrics"""
        if not self.results:
            return {}
        
        # Aggregate advanced metrics
        meteor_scores = []
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        bert_scores = []
        embedding_scores = []
        gpt4_correctness_scores = []
        gpt4_helpfulness_scores = []
        overall_scores = []
        
        for result in self.results:
            if "advanced_metrics" in result and result["advanced_metrics"] and "error" not in result["advanced_metrics"]:
                metrics = result["advanced_metrics"]
                meteor_scores.append(metrics.get("meteor_score", 0))
                rouge1_scores.append(metrics.get("rouge_scores", {}).get("rouge1", 0))
                rouge2_scores.append(metrics.get("rouge_scores", {}).get("rouge2", 0))
                rougeL_scores.append(metrics.get("rouge_scores", {}).get("rougeL", 0))
                bert_scores.append(metrics.get("bert_score", 0))
                embedding_scores.append(metrics.get("embedding_score", 0))
                gpt4_correctness_scores.append(metrics.get("gpt4_correctness", 0))
                gpt4_helpfulness_scores.append(metrics.get("gpt4_helpfulness", 0))
                overall_scores.append(metrics.get("overall_score", 0))
        
        return {
            "average_meteor_score": sum(meteor_scores) / len(meteor_scores) if meteor_scores else 0.0,
            "average_rouge1_score": sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0.0,
            "average_rouge2_score": sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0.0,
            "average_rougeL_score": sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0.0,
            "average_bert_score": sum(bert_scores) / len(bert_scores) if bert_scores else 0.0,
            "average_embedding_score": sum(embedding_scores) / len(embedding_scores) if embedding_scores else 0.0,
            "average_gpt4_correctness": sum(gpt4_correctness_scores) / len(gpt4_correctness_scores) if gpt4_correctness_scores else 0.0,
            "average_gpt4_helpfulness": sum(gpt4_helpfulness_scores) / len(gpt4_helpfulness_scores) if gpt4_helpfulness_scores else 0.0,
            "average_overall_score": sum(overall_scores) / len(overall_scores) if overall_scores else 0.0,
            "total_test_cases": len(self.results)
        }
    
    def _calculate_advanced_metrics_summary(self) -> Dict[str, Any]:
        """Calculate summary of advanced metrics"""
        if not self.results:
            return {}
        
        # Count successful evaluations
        successful_evaluations = sum(1 for result in self.results 
                                  if "advanced_metrics" in result and result["advanced_metrics"] 
                                  and "error" not in result["advanced_metrics"])
        
        # Calculate metric ranges
        all_meteor_scores = [r["advanced_metrics"].get("meteor_score", 0) for r in self.results 
                           if "advanced_metrics" in r and r["advanced_metrics"] and "error" not in r["advanced_metrics"]]
        all_overall_scores = [r["advanced_metrics"].get("overall_score", 0) for r in self.results 
                             if "advanced_metrics" in r and r["advanced_metrics"] and "error" not in r["advanced_metrics"]]
        
        return {
            "successful_evaluations": successful_evaluations,
            "total_evaluations": len(self.results),
            "success_rate": successful_evaluations / len(self.results) if self.results else 0.0,
            "meteor_score_range": {
                "min": min(all_meteor_scores) if all_meteor_scores else 0,
                "max": max(all_meteor_scores) if all_meteor_scores else 0,
                "std": np.std(all_meteor_scores) if all_meteor_scores else 0
            },
            "overall_score_range": {
                "min": min(all_overall_scores) if all_overall_scores else 0,
                "max": max(all_overall_scores) if all_overall_scores else 0,
                "std": np.std(all_overall_scores) if all_overall_scores else 0
            }
        }
    
    def generate_advanced_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive advanced evaluation report"""
        
        report = []
        report.append("=" * 80)
        report.append("ADVANCED RCAgent LLM Model Evaluation Report")
        report.append("=" * 80)
        report.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall metrics
        overall_metrics = results.get("overall_metrics", {})
        report.append("ADVANCED SEMANTIC METRICS:")
        report.append(f"  Average METEOR Score: {overall_metrics.get('average_meteor_score', 0):.3f}")
        report.append(f"  Average ROUGE-1 Score: {overall_metrics.get('average_rouge1_score', 0):.3f}")
        report.append(f"  Average ROUGE-2 Score: {overall_metrics.get('average_rouge2_score', 0):.3f}")
        report.append(f"  Average ROUGE-L Score: {overall_metrics.get('average_rougeL_score', 0):.3f}")
        report.append(f"  Average BERT Score: {overall_metrics.get('average_bert_score', 0):.3f}")
        report.append(f"  Average Embedding Score: {overall_metrics.get('average_embedding_score', 0):.3f}")
        report.append("")
        
        # GPT-4 metrics
        report.append("GPT-4 EVALUATION METRICS:")
        report.append(f"  Average G-Correctness: {overall_metrics.get('average_gpt4_correctness', 0):.1f}/10")
        report.append(f"  Average G-Helpfulness: {overall_metrics.get('average_gpt4_helpfulness', 0):.1f}/10")
        report.append("")
        
        # Overall performance
        report.append("OVERALL PERFORMANCE:")
        report.append(f"  Average Overall Score: {overall_metrics.get('average_overall_score', 0):.3f}")
        report.append(f"  Total Test Cases: {overall_metrics.get('total_test_cases', 0)}")
        report.append("")
        
        # Advanced metrics summary
        advanced_summary = results.get("advanced_metrics_summary", {})
        report.append("ADVANCED METRICS SUMMARY:")
        report.append(f"  Successful Evaluations: {advanced_summary.get('successful_evaluations', 0)}")
        report.append(f"  Success Rate: {advanced_summary.get('success_rate', 0):.1%}")
        report.append("")
        
        # Individual test case results
        report.append("INDIVIDUAL TEST CASE RESULTS:")
        for i, test_case in enumerate(results.get("test_cases", [])):
            report.append(f"\n--- Test Case {i+1} ---")
            report.append(f"Issue: {test_case['issue'][:100]}...")
            report.append(f"Analysis Time: {test_case['analysis_time']:.2f}s")
            
            if "advanced_metrics" in test_case and test_case["advanced_metrics"]:
                metrics = test_case["advanced_metrics"]
                if "error" not in metrics:
                    report.append(f"METEOR Score: {metrics.get('meteor_score', 0):.3f}")
                    report.append(f"ROUGE-1: {metrics.get('rouge_scores', {}).get('rouge1', 0):.3f}")
                    report.append(f"BERT Score: {metrics.get('bert_score', 0):.3f}")
                    report.append(f"Embedding Score: {metrics.get('embedding_score', 0):.3f}")
                    report.append(f"GPT-4 Correctness: {metrics.get('gpt4_correctness', 0):.1f}/10")
                    report.append(f"GPT-4 Helpfulness: {metrics.get('gpt4_helpfulness', 0):.1f}/10")
                    report.append(f"Overall Score: {metrics.get('overall_score', 0):.3f}")
                else:
                    report.append(f"Evaluation Error: {metrics['error']}")
        
        return "\n".join(report)

def create_advanced_test_cases() -> List[Dict[str, Any]]:
    """Create test cases for advanced evaluation"""
    
    return [
        {
            "issue": "High error rate in bank transfers",
            "files": [
                "C:\\Users\\rishi\\Downloads\\Bank\\telemetry\\2021_03_05\\log\\log_service.csv",
                "C:\\Users\\rishi\\Downloads\\Bank\\telemetry\\2021_03_05\\metric\\metric_app.csv",
                "C:\\Users\\rishi\\Downloads\\Bank\\telemetry\\2021_03_05\\trace\\trace_span.csv"
            ],
            "ground_truth": {
                "root_cause": "High latency in trace_span.csv causing performance issues in bank transfer operations",
                "recommendation": "Optimize database queries, implement caching, and increase connection pool size"
            }
        },
        {
            "issue": "Database connection timeouts",
            "files": [
                "C:\\Users\\rishi\\Downloads\\Bank\\telemetry\\2021_03_05\\metric\\metric_container.csv",
                "C:\\Users\\rishi\\Downloads\\Bank\\telemetry\\2021_03_05\\trace\\trace_span.csv"
            ],
            "ground_truth": {
                "root_cause": "Database connection pool exhaustion due to high concurrent requests",
                "recommendation": "Increase connection pool size, implement connection monitoring, and add retry logic"
            }
        }
    ]

def main():
    """Main advanced evaluation function"""
    
    print("Advanced RCAgent LLM Model Evaluation Suite")
    print("=" * 60)
    print("Using semantic metrics: METEOR, ROUGE, BERTScore, Embedding Similarity")
    print("Using GPT-4 evaluation: G-Correctness and G-Helpfulness")
    print("=" * 60)
    
    # Initialize evaluation suite
    # Note: You'll need to provide your OpenAI API key for GPT-4 evaluation
    evaluator = AdvancedRCAgentEvaluationSuite(
        model_name="llama3",
        openai_api_key=None  # Set your OpenAI API key here
    )
    
    # Create test cases
    test_cases = create_advanced_test_cases()
    
    # Run comprehensive evaluation
    results = evaluator.run_comprehensive_advanced_evaluation(test_cases)
    
    # Generate and save report
    report = evaluator.generate_advanced_evaluation_report(results)
    
    # Save results
    with open("advanced_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    with open("advanced_evaluation_report.txt", "w") as f:
        f.write(report)
    
    print("\nAdvanced evaluation completed!")
    print(f"Results saved to: advanced_evaluation_results.json")
    print(f"Report saved to: advanced_evaluation_report.txt")
    
    # Print summary
    print("\n" + "=" * 60)
    print("ADVANCED EVALUATION SUMMARY")
    print("=" * 60)
    print(report)

if __name__ == "__main__":
    main()

