"""
Run advanced evaluation without GPT-4 (using semantic metrics only)
"""

import json
import time
from advanced_evaluation_metrics import AdvancedRCAgentEvaluator
from rcagent_llm import RCAgentLLM

def run_advanced_evaluation_without_gpt4():
    """Run advanced evaluation using semantic metrics only"""
    
    print("Advanced RCAgent Evaluation (Semantic Metrics Only)")
    print("=" * 60)
    print("Using: METEOR, ROUGE, BERTScore, Embedding Similarity")
    print("=" * 60)
    
    # Initialize components
    rcagent = RCAgentLLM(model="llama3")
    evaluator = AdvancedRCAgentEvaluator()  # No OpenAI API key
    
    # Test case: Your Bank dataset
    issue = "High error rate in bank transfers"
    files = [
        "C:\\Users\\rishi\\Downloads\\Bank\\telemetry\\2021_03_05\\log\\log_service.csv",
        "C:\\Users\\rishi\\Downloads\\Bank\\telemetry\\2021_03_05\\metric\\metric_app.csv",
        "C:\\Users\\rishi\\Downloads\\Bank\\telemetry\\2021_03_05\\metric\\metric_container.csv",
        "C:\\Users\\rishi\\Downloads\\Bank\\telemetry\\2021_03_05\\trace\\trace_span.csv"
    ]
    
    # Ground truth for comparison
    ground_truth_root_cause = "High latency in trace_span.csv causing performance issues in bank transfer operations"
    ground_truth_recommendation = "Optimize database queries, implement caching, and increase connection pool size"
    
    print(f"Testing Issue: {issue}")
    print(f"Files: {len(files)} telemetry files")
    print()
    
    # Run RCAgent analysis
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
        predicted_root_cause = result_json.get("root_cause", "")
        predicted_recommendation = result_json.get("recommendation", "")
        
        print("TELEMETRY DATA:")
        print(f"  Files processed: {telemetry_data.get('files', 0)}")
        print(f"  Total rows: {telemetry_data.get('rows_total', 0):,}")
        print(f"  Error events: {telemetry_data.get('error_events_total', 0)}")
        print(f"  Worst latency P99: {telemetry_data.get('worst_latency_p99', 0)}ms")
        print()
        
        # Run advanced semantic evaluation
        print("ADVANCED SEMANTIC EVALUATION:")
        print("=" * 40)
        
        # Root cause evaluation
        print("ROOT CAUSE EVALUATION:")
        root_cause_scores = evaluator.evaluate_semantic_similarity(
            predicted_root_cause, ground_truth_root_cause
        )
        
        print(f"  METEOR Score: {root_cause_scores['meteor']:.3f}")
        print(f"  ROUGE-1: {root_cause_scores['rouge1']:.3f}")
        print(f"  ROUGE-2: {root_cause_scores['rouge2']:.3f}")
        print(f"  ROUGE-L: {root_cause_scores['rougeL']:.3f}")
        print(f"  BERT Score: {root_cause_scores['bert_score']:.3f}")
        print(f"  Embedding Score: {root_cause_scores['embedding_score']:.3f}")
        print()
        
        # Recommendation evaluation
        print("RECOMMENDATION EVALUATION:")
        recommendation_scores = evaluator.evaluate_semantic_similarity(
            predicted_recommendation, ground_truth_recommendation
        )
        
        print(f"  METEOR Score: {recommendation_scores['meteor']:.3f}")
        print(f"  ROUGE-1: {recommendation_scores['rouge1']:.3f}")
        print(f"  ROUGE-2: {recommendation_scores['rouge2']:.3f}")
        print(f"  ROUGE-L: {recommendation_scores['rougeL']:.3f}")
        print(f"  BERT Score: {recommendation_scores['bert_score']:.3f}")
        print(f"  Embedding Score: {recommendation_scores['embedding_score']:.3f}")
        print()
        
        # NUBIA metrics
        print("NUBIA 6-DIMENSIONAL EVALUATION:")
        nubia_root_cause = evaluator.evaluate_nubia_metrics(predicted_root_cause, ground_truth_root_cause)
        nubia_recommendation = evaluator.evaluate_nubia_metrics(predicted_recommendation, ground_truth_recommendation)
        
        print("Root Cause NUBIA Metrics:")
        for metric, score in nubia_root_cause.items():
            print(f"  {metric}: {score:.3f}")
        
        print("Recommendation NUBIA Metrics:")
        for metric, score in nubia_recommendation.items():
            print(f"  {metric}: {score:.3f}")
        print()
        
        # BARTScore metrics
        print("BARTScore EVALUATION:")
        bart_root_cause = evaluator.evaluate_bart_score(predicted_root_cause, ground_truth_root_cause)
        bart_recommendation = evaluator.evaluate_bart_score(predicted_recommendation, ground_truth_recommendation)
        
        print(f"  Root Cause F-Score: {bart_root_cause['f_score']:.3f}")
        print(f"  Root Cause CNN/DM: {bart_root_cause['cnn_dm']:.3f}")
        print(f"  Recommendation F-Score: {bart_recommendation['f_score']:.3f}")
        print(f"  Recommendation CNN/DM: {bart_recommendation['cnn_dm']:.3f}")
        print()
        
        # Overall evaluation
        print("OVERALL EVALUATION:")
        print(f"  Predicted Root Cause: {predicted_root_cause}")
        print(f"  Predicted Recommendation: {predicted_recommendation}")
        print()
        
        # Calculate overall scores
        avg_meteor = (root_cause_scores['meteor'] + recommendation_scores['meteor']) / 2
        avg_rouge1 = (root_cause_scores['rouge1'] + recommendation_scores['rouge1']) / 2
        avg_rouge2 = (root_cause_scores['rouge2'] + recommendation_scores['rouge2']) / 2
        avg_rougeL = (root_cause_scores['rougeL'] + recommendation_scores['rougeL']) / 2
        avg_bert = (root_cause_scores['bert_score'] + recommendation_scores['bert_score']) / 2
        avg_embedding = (root_cause_scores['embedding_score'] + recommendation_scores['embedding_score']) / 2
        
        overall_score = (avg_meteor + avg_rouge1 + avg_rouge2 + avg_rougeL + avg_bert + avg_embedding) / 6
        
        print("FINAL SCORES:")
        print(f"  Average METEOR: {avg_meteor:.3f}")
        print(f"  Average ROUGE-1: {avg_rouge1:.3f}")
        print(f"  Average ROUGE-2: {avg_rouge2:.3f}")
        print(f"  Average ROUGE-L: {avg_rougeL:.3f}")
        print(f"  Average BERT Score: {avg_bert:.3f}")
        print(f"  Average Embedding: {avg_embedding:.3f}")
        print(f"  Overall Score: {overall_score:.3f}")
        print()
        
        # Performance assessment
        if overall_score >= 0.8:
            print("Status: EXCELLENT - High quality analysis")
        elif overall_score >= 0.6:
            print("Status: GOOD - Solid analysis with room for improvement")
        elif overall_score >= 0.4:
            print("Status: FAIR - Analysis needs improvement")
        else:
            print("Status: POOR - Analysis quality is low")
        
        # Save results
        results = {
            "issue": issue,
            "analysis_time": analysis_time,
            "predicted_root_cause": predicted_root_cause,
            "predicted_recommendation": predicted_recommendation,
            "ground_truth_root_cause": ground_truth_root_cause,
            "ground_truth_recommendation": ground_truth_recommendation,
            "root_cause_scores": root_cause_scores,
            "recommendation_scores": recommendation_scores,
            "nubia_root_cause": nubia_root_cause,
            "nubia_recommendation": nubia_recommendation,
            "bart_root_cause": bart_root_cause,
            "bart_recommendation": bart_recommendation,
            "overall_score": overall_score,
            "telemetry_data": telemetry_data
        }
        
        with open("advanced_evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nResults saved to: advanced_evaluation_results.json")
        
        return results
        
    except Exception as e:
        print(f"Error during advanced evaluation: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    # Install required packages first:
    # pip install nltk rouge-score bert-score sentence-transformers scikit-learn
    
    print("Installing required packages...")
    print("Run: pip install nltk rouge-score bert-score sentence-transformers scikit-learn")
    print()
    
    # Run advanced evaluation
    results = run_advanced_evaluation_without_gpt4()
    
    print("\n" + "=" * 60)
    print("ADVANCED EVALUATION COMPLETE")
    print("=" * 60)
    
    if "error" not in results:
        print(f"Overall Score: {results.get('overall_score', 0):.3f}")
        print("\nTo add GPT-4 evaluation:")
        print("1. Get OpenAI API key from https://platform.openai.com/")
        print("2. Set openai_api_key in advanced_rcagent_evaluation.py")
        print("3. Run: python advanced_rcagent_evaluation.py")
    else:
        print(f"Evaluation failed: {results['error']}")

