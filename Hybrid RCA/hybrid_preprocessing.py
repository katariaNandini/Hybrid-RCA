import os
import json
from pathlib import Path
from typing import Dict, Any, List, Tuple


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_metric_app(metric_app_path: Path) -> Dict[str, Dict[str, List[float]]]:
    """
    Reads metric_app.csv with header:
      timestamp,rr,sr,cnt,mrt,tc
    Returns per-service arrays for mrt (latency), sr (success), cnt (throughput), rr.
    """
    per_service: Dict[str, Dict[str, List[float]]] = {}
    with metric_app_path.open("r", encoding="utf-8", errors="ignore") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 6:
                continue
            # timestamp = int(parts[0])  # unused here
            try:
                rr = float(parts[1])
                sr = float(parts[2])
                cnt = float(parts[3])
                mrt = float(parts[4])
            except ValueError:
                continue
            svc = parts[5]
            bucket = per_service.setdefault(svc, {"mrt": [], "sr": [], "cnt": [], "rr": []})
            bucket["mrt"].append(mrt)
            bucket["sr"].append(sr)
            bucket["cnt"].append(cnt)
            bucket["rr"].append(rr)
    return per_service


def _percentile(sorted_vals: List[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return d0 + d1


def _aggregate_service_stats(per_service: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    agg: Dict[str, Dict[str, float]] = {}
    for svc, series in per_service.items():
        mrt_vals = sorted(series.get("mrt", []))
        sr_vals = series.get("sr", [])
        cnt_vals = series.get("cnt", [])
        rr_vals = series.get("rr", [])
        p50 = _percentile(mrt_vals, 0.5)
        p95 = _percentile(mrt_vals, 0.95)
        p99 = _percentile(mrt_vals, 0.99)
        avg_sr = sum(sr_vals) / len(sr_vals) if sr_vals else 0.0
        avg_cnt = sum(cnt_vals) / len(cnt_vals) if cnt_vals else 0.0
        avg_rr = sum(rr_vals) / len(rr_vals) if rr_vals else 0.0
        agg[svc] = {
            "latency_p50": p50,
            "latency_p95": p95,
            "latency_p99": p99,
            "success_rate": avg_sr,
            "error_rate": max(0.0, 100.0 - avg_sr),
            "throughput": avg_cnt,
            "request_rate": avg_rr,
        }
    return agg


def _detect_issues(service_stats: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
    if not service_stats:
        return []
    p95_all = [v["latency_p95"] for v in service_stats.values()]
    p95_all_sorted = sorted(p95_all)
    global_p95_med = _percentile(p95_all_sorted, 0.5)
    issues: List[Dict[str, Any]] = []
    for svc, stats in service_stats.items():
        lat_spike = stats["latency_p95"] > (1.5 * global_p95_med if global_p95_med > 0 else stats["latency_p95"]) and stats["latency_p95"] > 0
        err_burst = stats["success_rate"] < 99.0
        score = 0.0
        reasons: List[str] = []
        if lat_spike:
            score += 0.6
            reasons.append("p95_latency_spike")
        if err_burst:
            score += 0.4
            reasons.append("success_rate_drop")
        if score > 0:
            issues.append({"service": svc, "type": ",".join(reasons), "score": round(min(score, 1.0), 3)})
    # sort by score desc
    issues.sort(key=lambda x: x["score"], reverse=True)
    return issues


def preprocess_session(session_dir: str, config_path: str | None = None) -> Dict[str, Any]:
    """
    Preprocess telemetry session:
      - parse metric_app.csv (timestamp,rr,sr,cnt,mrt,tc)
      - compute per-service aggregates
      - detect issues from latency and success rate
      - emit placeholder trace edges (trace is huge; integrate later)
    """
    session_path = Path(session_dir)
    session_name = session_path.name

    if not session_path.exists():
        alt = Path("KGroot") / "Bank" / "telemetry" / session_name
        if alt.exists():
            session_path = alt
        else:
            raise FileNotFoundError(f"Session directory not found: {session_dir}")

    metrics_dir = session_path / "metric"
    metric_app = metrics_dir / "metric_app.csv"

    output_root = Path("Hybrid RCA") / "outputs" / session_name
    _ensure_dir(output_root)

    per_service = {}
    if metric_app.exists():
        per_service = _read_metric_app(metric_app)
    service_stats = _aggregate_service_stats(per_service) if per_service else {}

    processed_metrics = {
        "session": session_name,
        "services": service_stats,
    }
    (output_root / "processed_metrics.parquet").write_text(json.dumps(processed_metrics, indent=2))

    # Logs/traces placeholders (huge files not parsed in this step)
    (output_root / "processed_logs.parquet").write_text(json.dumps({"session": session_name}))
    (output_root / "processed_traces.parquet").write_text(json.dumps({"session": session_name}))

    # Build simple co-anomaly edges: services with high score connected sequentially
    issues = _detect_issues(service_stats)
    if issues:
        top = [i["service"] for i in issues[:3]]
    else:
        top = [svc for svc in list(service_stats.keys())[:3]]
    edges_lines = ["src,dst,weight"]
    for i in range(len(top) - 1):
        edges_lines.append(f"{top[i]},{top[i+1]},0.7")
    if len(top) >= 2:
        edges_lines.append(f"frontend,{top[0]},0.6")
    (output_root / "trace_edges.csv").write_text("\n".join(edges_lines) + "\n")

    issues_payload = {"session": session_name, "issues": issues}
    (output_root / "issues.json").write_text(json.dumps(issues_payload, indent=2))

    return {
        "session": session_name,
        "num_services": len(service_stats),
        "output_dir": str(output_root),
    }
