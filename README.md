Hybrid RCA
==========

Quick start
-----------

1) Python 3.10+ recommended. Create and activate a virtual environment.

```bash
python -m venv .venv
.venv\\Scripts\\activate  # Windows PowerShell
```

2) Install dependencies (aggregated from sub-projects).

```bash
pip install -r requirements.txt
```

3) Run the hybrid pipeline with the included sample bank telemetry.

```bash
python "Hybrid RCA\\run_hybrid.py"
```

Project layout
--------------

- `Hybrid RCA/` main hybrid orchestrator and preprocessing scripts
- `KGroot/` knowledge-graph root cause library and sample datasets
- `RCAgent/` LLM-based RCA agent and evaluation tools
- `Hybrid RCA/outputs/` example outputs for a sample date (kept for reproducibility)

Notes
-----

- Large or derived artifacts (e.g., `RCAgent/venv/`, `KGroot/runs/`) remain ignored.
- If the repository size grows too large, consider re-ignoring `outputs/` or using Git LFS for big files.

