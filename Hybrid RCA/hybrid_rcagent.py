import json
from pathlib import Path
from typing import Dict, Any, List


def _read_issues(output_dir: Path) -> List[Dict[str, Any]]:
    issues_path = output_dir / "issues.json"
    if not issues_path.exists():
        return []
    data = json.loads(issues_path.read_text())
    return data.get("issues", [])


def run_rcagent(session_dir: str, mode: str = "offline") -> Dict[str, Any]:
    """
    Wrapper for RCAgent module. Stubbed logic:
      - Offline mode: rank services based on placeholder issues.json
      - LLM mode: same stub, with mode tag

    Expects preprocessing to have created outputs at: Hybrid RCA/outputs/<session>/
    """
    session_name = Path(session_dir).name
    output_dir = Path("Hybrid RCA") / "outputs" / session_name

    issues = _read_issues(output_dir)
    # naive ranking by score
    candidates = [
        {"service": it["service"], "score": float(it.get("score", 0.5))}
        for it in issues
    ]
    candidates.sort(key=lambda x: x["score"], reverse=True)

    causal_graph = {
        "nodes": ["frontend", "cart", "checkout", "payment"],
        "edges": [
            {"src": "frontend", "dst": "checkout", "weight": 0.6},
            {"src": "checkout", "dst": "payment", "weight": 0.7},
        ],
    }

    recommendations = [
        "Review DB pool settings in checkout",
        "Inspect payment provider timeout thresholds",
    ]

    rc_result = {
        "mode": mode,
        "candidates": candidates or [
            {"service": "checkout", "score": 0.6},
            {"service": "payment", "score": 0.55},
        ],
        "causal_graph": causal_graph,
        "recommendations": recommendations,
    }

    (output_dir / "rcagent_output.json").write_text(json.dumps(rc_result, indent=2))
    (output_dir / "causal_graph.json").write_text(json.dumps(causal_graph, indent=2))

    return rc_result
