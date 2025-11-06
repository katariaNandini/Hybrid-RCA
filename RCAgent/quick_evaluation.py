"""
Quick evaluation script for RCAgent LLM model
Run this to evaluate your current RCAgent setup
"""

import json
import time
from rcagent_llm import RCAgentLLM
from evaluation_metrics import RCAgentEvaluator

def quick_evaluate_rcagent():
    """Quick evaluation of RCAgent performance"""
    
    print("RCAgent LLM Model Quick Evaluation")
    print("=" * 50)
    
    # Initialize components
    rcagent = RCAgentLLM(model="llama3")
    evaluator = RCAgentEvaluator()
    
    # Test case: Your Bank dataset
    issue = "High error rate in bank transfers"
    files = [
        "C:\\Users\\rishi\\Downloads\\Bank\\telemetry\\2021_03_05\\log\\log_service.csv",
        "C:\\Users\\rishi\\Downloads\\Bank\\telemetry\\2021_03_05\\metric\\metric_app.csv",
        "C:\\Users\\rishi\\Downloads\\Bank\\telemetry\\2021_03_05\\metric\\metric_container.csv",
        "C:\\Users\\rishi\\Downloads\\Bank\\telemetry\\2021_03_05\\trace\\trace_span.csv"
    ]
    
    print(f"Testing Issue: {issue}")
    print(f"Files: {len(files)} telemetry files")
    print()
    
    # Run analysis
    print("Running RCAgent analysis...")
    start_time = time.time()
    
    try:
        result = rcagent.analyze(issue, files)
        analysis_time = time.time() - start_time
        
        # Parse result
        if isinstance(result, str):
            result_json = json.loads(result)
        else:
            result_json = result
            
        print(f"Analysis completed in {analysis_time:.2f} seconds")
        print()
        
        # Extract telemetry data
        telemetry_data = result_json.get("summary", {})
        root_cause = result_json.get("root_cause", "")
        recommendation = result_json.get("recommendation", "")
        
        print("TELEMETRY DATA:")
        print(f"  Files processed: {telemetry_data.get('files', 0)}")
        print(f"  Total rows: {telemetry_data.get('rows_total', 0):,}")
        print(f"  Error events: {telemetry_data.get('error_events_total', 0)}")
        print(f"  Worst latency P99: {telemetry_data.get('worst_latency_p99', 0)}ms")
        print()
        
        # Evaluate confidence metrics
        print("CONFIDENCE EVALUATION:")
        confidence_metrics = evaluator.calculate_confidence_metrics(
            json.dumps(result_json), 
            telemetry_data
        )
        
        print(f"  Overall Confidence: {confidence_metrics['overall_confidence']:.3f}")
        print(f"  Data Coverage: {confidence_metrics['data_coverage_confidence']:.3f}")
        print(f"  Technical Confidence: {confidence_metrics['technical_confidence']:.3f}")
        print(f"  Consistency Score: {confidence_metrics['consistency_score']:.3f}")
        print(f"  Uncertainty Score: {confidence_metrics['uncertainty_score']:.3f}")
        print()
        
        # Evaluate root cause quality
        print("ROOT CAUSE ANALYSIS:")
        print(f"  Root Cause: {root_cause}")
        
        # Evaluate recommendation quality
        print("RECOMMENDATION ANALYSIS:")
        print(f"  Recommendation: {recommendation}")
        
        # Calculate quality scores
        root_cause_quality = evaluator._calculate_specificity_score(root_cause)
        recommendation_quality = evaluator._calculate_actionability_score(recommendation)
        
        print()
        print("QUALITY METRICS:")
        print(f"  Root Cause Specificity: {root_cause_quality:.3f}")
        print(f"  Recommendation Actionability: {recommendation_quality:.3f}")
        print()
        
        # Overall evaluation
        overall_score = (
            confidence_metrics['overall_confidence'] + 
            root_cause_quality + 
            recommendation_quality
        ) / 3
        
        print("OVERALL EVALUATION:")
        print(f"  Overall Score: {overall_score:.3f}")
        
        if overall_score >= 0.8:
            print("  Status: EXCELLENT - High quality analysis")
        elif overall_score >= 0.6:
            print("  Status: GOOD - Solid analysis with room for improvement")
        elif overall_score >= 0.4:
            print("  Status: FAIR - Analysis needs improvement")
        else:
            print("  Status: POOR - Analysis quality is low")
        
        print()
        print("DETAILED RESULTS:")
        print(json.dumps(result_json, indent=2))
        
        return {
            "overall_score": overall_score,
            "confidence_metrics": confidence_metrics,
            "root_cause_quality": root_cause_quality,
            "recommendation_quality": recommendation_quality,
            "analysis_time": analysis_time,
            "result": result_json
        }
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return {"error": str(e)}

def run_consistency_test():
    """Test consistency across multiple runs"""
    
    print("\nCONSISTENCY TEST")
    print("=" * 30)
    
    rcagent = RCAgentLLM(model="llama3")
    evaluator = RCAgentEvaluator()
    
    issue = "High error rate in bank transfers"
    files = [
        "C:\\Users\\rishi\\Downloads\\Bank\\telemetry\\2021_03_05\\trace\\trace_span.csv"
    ]
    
    print("Running 3 consistency tests...")
    
    responses = []
    for i in range(3):
        print(f"  Run {i+1}/3...")
        try:
            result = rcagent.analyze(issue, files)
            if isinstance(result, str):
                result_json = json.loads(result)
            else:
                result_json = result
            responses.append(json.dumps(result_json))
        except Exception as e:
            print(f"    Error: {e}")
            responses.append(f"Error: {e}")
    
    # Calculate consistency
    consistency_score = evaluator.evaluate_consistency(responses)
    
    print(f"Consistency Score: {consistency_score:.3f}")
    
    if consistency_score >= 0.8:
        print("Status: HIGH CONSISTENCY")
    elif consistency_score >= 0.6:
        print("Status: MODERATE CONSISTENCY")
    else:
        print("Status: LOW CONSISTENCY")
    
    return consistency_score

if __name__ == "__main__":
    # Run quick evaluation
    results = quick_evaluate_rcagent()
    
    # Run consistency test
    consistency = run_consistency_test()
    
    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE")
    print("=" * 50)
    print(f"Overall Score: {results.get('overall_score', 0):.3f}")
    print(f"Consistency Score: {consistency:.3f}")
    print("\nTo improve performance:")
    print("  - Ensure telemetry data is complete and accurate")
    print("  - Use specific, detailed issue descriptions")
    print("  - Include relevant metrics in the analysis")
    print("  - Consider fine-tuning the LLM model for your domain")
