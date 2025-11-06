import os
import json
import subprocess
import argparse
from typing import List, Dict, Any
from urllib.parse import urlparse
import requests

import pandas as pd

SNAPSHOT_DIR = os.getenv("RCAGENT_SNAPSHOT_DIR", "snapshots")


class RCAgentLLM:
    def __init__(self, model: str = "llama3"):
        self.model = model

    def _call_ollama(self, prompt: str) -> str:
        result = subprocess.run(
            ["ollama", "run", self.model],
            input=prompt.encode(),
            capture_output=True,
            check=True,
        )
        return result.stdout.decode().strip()

    # -------------------- OFFLINE ANALYSIS (no LLM) --------------------
    def _read_csv_safe(self, path: str) -> pd.DataFrame:
        """Read CSV from local file or Google Drive URL."""
        try:
            # Check if it's a URL
            if path.startswith(('http://', 'https://')):
                return self._read_csv_from_url(path)
            else:
                return pd.read_csv(path)
        except Exception:
            # Retry with common options for messy CSVs
            try:
                if path.startswith(('http://', 'https://')):
                    return self._read_csv_from_url(path, encoding_errors="ignore", on_bad_lines="skip")
                else:
                    return pd.read_csv(path, encoding_errors="ignore", on_bad_lines="skip")
            except Exception:
                return pd.DataFrame()

    def _read_csv_from_url(self, url: str, **kwargs) -> pd.DataFrame:
        """Read CSV from URL, handling Google Drive sharing links."""
        # Convert Google Drive sharing URL to direct download URL
        if 'drive.google.com' in url:
            file_id = self._extract_google_drive_file_id(url)
            if file_id:
                url = f"https://drive.google.com/uc?export=download&id={file_id}"
        
        # Download and read CSV
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Use StringIO to read from memory
        from io import StringIO
        csv_content = StringIO(response.text)
        return pd.read_csv(csv_content, **kwargs)

    def _extract_google_drive_file_id(self, url: str) -> str:
        """Extract file ID from Google Drive sharing URL."""
        # Handle different Google Drive URL formats
        if '/file/d/' in url:
            # Format: https://drive.google.com/file/d/FILE_ID/view
            start = url.find('/file/d/') + 8
            end = url.find('/', start)
            if end == -1:
                end = url.find('?', start)
            return url[start:end] if end > start else None
        elif 'id=' in url:
            # Format: https://drive.google.com/open?id=FILE_ID
            start = url.find('id=') + 3
            end = url.find('&', start)
            if end == -1:
                end = len(url)
            return url[start:end]
        return None

    def _summarize_file(self, path: str) -> Dict[str, Any]:
        summary: Dict[str, Any] = {"path": path}
        # Check if it's a CSV file (local or URL)
        is_csv = path.lower().endswith(".csv")
        is_local = os.path.exists(path) if not path.startswith(('http://', 'https://')) else True
        
        if is_csv and is_local:
            df = self._read_csv_safe(path)
            summary["rows"] = int(df.shape[0])
            summary["cols"] = int(df.shape[1])
            summary["columns"] = list(df.columns.astype(str))

            # Basic error/level/status indicators
            indicators: Dict[str, Any] = {}
            colnames = {c.lower(): c for c in df.columns}

            # Error-like columns
            error_cols = [name for name in colnames if any(k in name for k in ["error", "exception", "fail"]) ]
            error_count = 0
            for lc in error_cols:
                c = colnames[lc]
                series = df[c].astype(str).str.lower()
                error_count += int(series.isin(["true", "1", "yes"]).sum())
                error_count += int(series.str.contains("error|exception|fail", na=False).sum())
            indicators["error_events"] = int(error_count)

            # Status/level columns
            status_cols = [name for name in colnames if any(k in name for k in ["status", "code", "http_status"]) ]
            level_cols = [name for name in colnames if "level" in name]
            status_breakdown: Dict[str, int] = {}
            for lc in status_cols + level_cols:
                c = colnames[lc]
                vc = df[c].astype(str).value_counts(dropna=False).head(10)
                status_breakdown[c] = {str(k): int(v) for k, v in vc.items()}
            indicators["status_level"] = status_breakdown

            # Latency/duration columns
            latency_cols = [name for name in colnames if any(k in name for k in ["latency", "duration", "elapsed"]) ]
            latency_stats: Dict[str, Any] = {}
            for lc in latency_cols:
                c = colnames[lc]
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().any():
                    latency_stats[c] = {
                        "p50": float(s.quantile(0.50)),
                        "p90": float(s.quantile(0.90)),
                        "p99": float(s.quantile(0.99)),
                        "max": float(s.max()),
                        "count": int(s.count()),
                    }
            indicators["latency"] = latency_stats

            summary["indicators"] = indicators
        else:
            summary["note"] = "Unsupported or missing file"
        return summary

    def offline_analyze(self, issue: str, snapshot_files: List[str]) -> str:
        # Resolve relative files via SNAPSHOT_DIR
        resolved_paths = [f if os.path.isabs(f) else os.path.join(SNAPSHOT_DIR, f) for f in snapshot_files]
        summaries = [self._summarize_file(p) for p in resolved_paths]

        # Aggregate simple signals
        total_rows = sum(s.get("rows", 0) for s in summaries)
        total_errors = sum(s.get("indicators", {}).get("error_events", 0) for s in summaries)

        # Find worst latency p99 across files
        worst_latency = 0.0
        worst_file = None
        for s in summaries:
            lat = s.get("indicators", {}).get("latency", {})
            for col, stats in lat.items():
                p99 = float(stats.get("p99", 0.0))
                if p99 > worst_latency:
                    worst_latency = p99
                    worst_file = s.get("path")

        # Heuristic root cause
        if total_errors > 0 and worst_latency > 0:
            root_cause = (
                "Elevated error events observed alongside high tail latency; likely downstream dependency issues or resource saturation."
            )
            recommendation = (
                "Investigate services emitting errors, check recent deploys, and examine resource limits for pods/containers. Enable retries and add latency budgets."
            )
        elif total_errors > 0:
            root_cause = "Increased error events detected across telemetry files."
            recommendation = "Inspect error logs for dominant failure codes, roll back recent changes, add alerts on error spikes."
        elif worst_latency > 0:
            root_cause = f"High tail latency detected (p99≈{worst_latency:.2f}) in {worst_file or 'unknown source'}."
            recommendation = "Profile hot paths, verify DB/index performance, and autoscale based on latency."
        else:
            root_cause = "No strong error or latency signals detected from provided snapshots."
            recommendation = "Expand time window and include additional metrics (CPU, memory, I/O) for correlation."

        result: Dict[str, Any] = {
            "issue": issue,
            "summary": {
                "files": len(summaries),
                "rows_total": total_rows,
                "error_events_total": total_errors,
                "worst_latency_p99": worst_latency,
                "worst_latency_file": worst_file,
            },
            "root_cause": root_cause,
            "recommendation": recommendation,
            "details": summaries[:10],  # keep output bounded
        }
        return json.dumps(result, indent=2)

    def load_snapshot(self, filename: str) -> str:
        """Load snapshot from snapshots/ folder or absolute path and return as string."""
        # Allow absolute paths; otherwise resolve relative to SNAPSHOT_DIR
        path = filename if os.path.isabs(filename) else os.path.join(SNAPSHOT_DIR, filename)
        if not os.path.exists(path):
            return f"[Snapshot not found: {filename}]"
        try:
            with open(path) as f:
                data = json.load(f)
            return json.dumps(data, indent=2)[:4000]  # truncate long JSON
        except Exception:
            with open(path) as f:
                return f.read()[:4000]

    def analyze(self, issue: str, snapshot_files: List[str]) -> str:
        """Combine issue + snapshot data, include structured telemetry summary, and query LLM."""
        # Resolve paths for consistent loading
        resolved_paths = [f if os.path.isabs(f) else os.path.join(SNAPSHOT_DIR, f) for f in snapshot_files]

        # Raw content (truncated) for non-CSV files and lightweight CSV preview
        raw_context_parts: List[str] = []
        summaries: List[Dict[str, Any]] = []
        for p in resolved_paths:
            if p.lower().endswith(".csv"):
                summaries.append(self._summarize_file(p))
                # Also include a tiny CSV head preview to help grounding
                try:
                    df_head = self._read_csv_safe(p).head(5)
                    raw_context_parts.append(f"[CSV HEAD] {p}\n" + df_head.to_csv(index=False)[:1000])
                except Exception:
                    pass
            else:
                # Use existing loader for text/json
                display_name = p if os.path.isabs(p) else os.path.relpath(p, SNAPSHOT_DIR)
                raw_context_parts.append(f"[FILE] {display_name}\n" + self.load_snapshot(p))

        # Aggregate headline metrics similar to offline mode
        total_rows = sum(s.get("rows", 0) for s in summaries)
        total_errors = sum(s.get("indicators", {}).get("error_events", 0) for s in summaries)
        worst_latency = 0.0
        worst_file = None
        for s in summaries:
            lat = s.get("indicators", {}).get("latency", {})
            for _, stats in lat.items():
                p99 = float(stats.get("p99", 0.0))
                if p99 > worst_latency:
                    worst_latency = p99
                    worst_file = s.get("path")

        telemetry_summary = {
            "files": len(summaries),
            "rows_total": total_rows,
            "error_events_total": total_errors,
            "worst_latency_p99": worst_latency,
            "worst_latency_file": worst_file,
            "files_summaries": summaries[:8],
        }

        raw_context = "\n\n--- SNAPSHOT ---\n\n".join(raw_context_parts)
        prompt = (
            "You are a production RCA (Root Cause Analysis) expert. Analyze the issue using the telemetry data.\n\n"
            f"ISSUE:\n{issue}\n\n"
            "TELEMETRY_SUMMARY (JSON):\n"
            f"{json.dumps(telemetry_summary, indent=2)}\n\n"
            "RAW_CONTEXT (truncated for brevity):\n"
            f"{raw_context}\n\n"
            "Instructions:\n"
            "- Use the telemetry_summary data to provide specific analysis.\n"
            "- Include actual numbers from the data (rows_total, error_events_total, worst_latency_p99, etc.).\n"
            "- Determine a concrete root cause based on the metrics.\n"
            "- Provide a specific recommendation.\n"
            "- Do NOT use placeholders like '...' or vague terms.\n"
            "- Output ONLY a simple JSON object with exactly these two keys: root_cause, recommendation.\n"
            "- Keep root_cause and recommendation under 400 characters each.\n"
            "- Do NOT include any other fields or nested structures.\n\n"
            "Example format:\n"
            "{\n"
            '  "root_cause": "Specific issue identified from the data",\n'
            '  "recommendation": "Specific action to take"\n'
            "}\n\n"
            "Return ONLY the JSON object with no extra text."
        )
        llm_response = self._call_ollama(prompt)
        
        # Try to parse the LLM response and merge with our structured data
        try:
            llm_json = json.loads(llm_response)
            # Create the complete structured response
            complete_response = {
                "issue": issue,
                "summary": telemetry_summary,
                "root_cause": llm_json.get("root_cause", "Analysis incomplete"),
                "recommendation": llm_json.get("recommendation", "No recommendation provided"),
                "details": summaries[:10]  # keep output bounded
            }
            return json.dumps(complete_response, indent=2)
        except json.JSONDecodeError:
            # Fallback: create structured response with LLM text
            complete_response = {
                "issue": issue,
                "summary": telemetry_summary,
                "root_cause": llm_response,
                "recommendation": "LLM response could not be parsed",
                "details": summaries[:10]
            }
            return json.dumps(complete_response, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RCAgent analysis with snapshots")
    parser.add_argument("--issue", required=False, default="High service latency observed in checkout API", help="Issue description for analysis")
    parser.add_argument("--model", required=False, default=os.getenv("RCAGENT_MODEL", "llama3"), help="Ollama model name")
    parser.add_argument("--snapshot-dir", required=False, default=os.getenv("RCAGENT_SNAPSHOT_DIR", SNAPSHOT_DIR), help="Directory containing snapshot files (for relative paths)")
    parser.add_argument("--files", nargs="+", required=False, default=["prometheus.json", "loki.json", "jaeger.txt"], help="List of snapshot files (relative to snapshot-dir) or absolute paths")
    parser.add_argument("--offline", action="store_true", help="Run offline analysis without Ollama (CSV heuristics)")

    args = parser.parse_args()

    # Override snapshot directory for relative file resolution
    SNAPSHOT_DIR = args.snapshot_dir

    agent = RCAgentLLM(model=args.model)
    if args.offline:
        result = agent.offline_analyze(issue=args.issue, snapshot_files=args.files)
    else:
        try:
            result = agent.analyze(issue=args.issue, snapshot_files=args.files)
        except FileNotFoundError:
            # ollama not available; fall back to offline
            result = agent.offline_analyze(issue=args.issue, snapshot_files=args.files)
    print(result)
