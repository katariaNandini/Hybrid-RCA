import json
from pathlib import Path
from typing import Dict, Any

from hybrid_preprocessing import preprocess_session
from hybrid_rcagent import run_rcagent
from hybrid_kgroot import run_kgroot


def _compute_weights(kg_sim: float, coverage: float) -> tuple[float, float]:
    if kg_sim >= 0.75 and coverage >= 0.8:
        return 0.4, 0.6  # w1 (rcagent), w2 (kgroot)
    return 0.6, 0.4


def run_hybrid(session_dir: str, rcagent_mode: str, kgroot_config: str | None, config_path: str | None) -> Dict[str, Any]:
    session_name = Path(session_dir).name
    output_dir = Path("Hybrid RCA") / "outputs" / session_name
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocess_info = preprocess_session(session_dir, config_path)
    rc = run_rcagent(session_dir, mode=rcagent_mode)
    kg = run_kgroot(session_dir, config_path=kgroot_config)

    kg_sim = float(kg.get("graph_similarity", 0.0))
    coverage = float(kg.get("node_feature_coverage", 0.0))
    w1, w2 = _compute_weights(kg_sim, coverage)

    # Late fusion of scores (align by service name)
    rc_scores = {c["service"]: float(c["score"]) for c in rc.get("candidates", [])}
    kg_scores = {c["service"]: float(c["score"]) for c in kg.get("candidates", [])}
    services = set(rc_scores) | set(kg_scores)
    fused = []
    for svc in services:
        s1 = rc_scores.get(svc, 0.0)
        s2 = kg_scores.get(svc, 0.0)
        fused.append({"service": svc, "score": w1 * s1 + w2 * s2})
    fused.sort(key=lambda x: x["score"], reverse=True)

    explanation = {
        "rcagent": {
            "mode": rc.get("mode"),
            "top": rc.get("candidates", [])[:3],
        },
        "kgroot": {
            "graph_similarity": kg_sim,
            "node_feature_coverage": coverage,
            "top": kg.get("candidates", [])[:3],
        },
    }

    result = {
        "session": session_name,
        "weights": {"rcagent": w1, "kgroot": w2},
        "fused_candidates": fused,
        "explanation": explanation,
    }

    (output_dir / "hybrid_result.json").write_text(json.dumps(result, indent=2))
    # Human-readable report
    report = [
        f"Hybrid RCA Report - Session {session_name}",
        f"Weights -> RCAgent: {w1:.2f}, KGroot: {w2:.2f}",
        f"KG Similarity: {kg_sim:.2f}, Coverage: {coverage:.2f}",
        "Top fused candidates:",
    ]
    for i, c in enumerate(result["fused_candidates"][:5], 1):
        report.append(f"  {i}. {c['service']} (score {c['score']:.2f})")
    (output_dir / "report.md").write_text("\n".join(report))

    return result
