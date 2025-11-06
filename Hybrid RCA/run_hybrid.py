import argparse
from pathlib import Path

# Allow running as a script without installing package
import sys
sys.path.append(str(Path(__file__).resolve().parent))

from hybrid_orchestrator import run_hybrid  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Hybrid RCA pipeline")
    parser.add_argument("--session-dir", required=True, help="Telemetry session directory or date name")
    parser.add_argument("--rcagent-mode", default="offline", choices=["offline", "llm"], help="RCAgent mode")
    parser.add_argument("--kgroot-config", default="KGroot/config_graph_sim.ini", help="KGroot config path")
    parser.add_argument("--config", default="Hybrid RCA/hybrid_config.yaml", help="Hybrid config path")
    parser.add_argument("--top-k", type=int, default=3, help="Top-k to display")
    args = parser.parse_args()

    session_dir = args.session_dir
    # Accept bare date like 2021_03_24
    if not Path(session_dir).exists():
        maybe = Path("KGroot") / "Bank" / "telemetry" / session_dir
        if maybe.exists():
            session_dir = str(maybe)

    result = run_hybrid(session_dir, args.rcagent_mode, args.kgroot_config, args.config)
    top_k = result.get("fused_candidates", [])[: args.top_k]
    print("Top-{} candidates:".format(args.top_k))
    for i, c in enumerate(top_k, 1):
        print(f"  {i}. {c['service']} (score {c['score']:.2f})")


if __name__ == "__main__":
    main()
