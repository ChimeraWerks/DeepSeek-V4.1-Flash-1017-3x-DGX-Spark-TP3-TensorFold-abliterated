"""One install/start entry point and explicit owned-service operations."""
import argparse
import json
import os
import subprocess
import sys

from . import lifecycle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["up", "serve", "status", "stop", "rollback", "plan", "supervise", "prepare-index"])
    parser.add_argument("--config")
    parser.add_argument("--snapshot")
    parser.add_argument("--workdir")
    parser.add_argument("--with-engine", action="store_true")
    parser.add_argument("--qualification-window", action="store_true")
    args = parser.parse_args()
    try:
        if args.action == "prepare-index":
            if not args.snapshot or not args.workdir:
                parser.error("prepare-index requires --snapshot and --workdir")
            from .prepare import writable_index
            print(json.dumps({"index": str(writable_index(args.snapshot, args.workdir))}))
            return
        if not args.config:
            parser.error("--config is required")
        if args.qualification_window:
            os.environ["CHIMERA_QUALIFICATION_WINDOW"] = "1"
        path, raw, cfg = lifecycle.document(args.config)
        if args.action == "supervise":
            from .supervisor import supervise
            raise SystemExit(supervise(path))
        if args.action == "serve":
            lifecycle.load_env(path, raw)
            import uvicorn
            from .api import create_app
            uvicorn.run(create_app(cfg), host=cfg.host, port=cfg.port, workers=1, access_log=False)
            return
        if args.action == "up":
            result = lifecycle.up(path)
        elif args.action == "rollback":
            result = lifecycle.rollback(path)
        elif args.action == "plan":
            result = {"api_unit": lifecycle.unit_name(raw), "api": cfg.public(),
                      "native": lifecycle.rank_plan(raw["cluster"]) if raw.get("cluster") else "attach-existing"}
        elif args.action == "stop":
            result = lifecycle.stop_installation(path, args.with_engine)
        else:
            result = {"unit": lifecycle.unit_name(raw), "state": subprocess.run(
                ["systemctl", "--user", "is-active", lifecycle.unit_name(raw)], text=True, capture_output=True).stdout.strip()}
        print(json.dumps(result, indent=2))
    except (ValueError, RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        # Avoid echoing subprocess arguments or credential-bearing file content.
        print("PROBLEM: " + (str(exc) if not isinstance(exc, subprocess.CalledProcessError)
                              else "owned service command failed; inspect its user journal"), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
