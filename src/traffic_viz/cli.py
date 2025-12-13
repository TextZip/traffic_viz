from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path
import importlib.util


from traffic_viz.config import get_data_dir
from traffic_viz.preprocess_5min import preprocess_dir


def _has_parquet_cache(cache_dir: Path) -> bool:
    return cache_dir.exists() and any(cache_dir.glob("*.parquet"))


def _has_station_metadata(meta_dir: Path) -> bool:
    return meta_dir.exists() and any(meta_dir.glob("d*_text_meta_*.txt"))


def _has_raw_5min(raw_dir: Path) -> bool:
    return raw_dir.exists() and (any(raw_dir.glob("*.txt.gz")) or any(raw_dir.glob("*.txt")))


def run_streamlit_app() -> None:
    # Locate the installed module file path for traffic_viz.app
    spec = importlib.util.find_spec("traffic_viz.app")
    if spec is None or spec.origin is None:
        raise RuntimeError(
            "Could not locate module traffic_viz.app (is the package installed?)")

    app_path = Path(spec.origin).resolve()

    # streamlit expects a filepath, not `-m module`
    subprocess.run(["streamlit", "run", str(app_path)], check=True)


def main():
    p = argparse.ArgumentParser(prog="traffic_viz")
    sub = p.add_subparsers(dest="cmd", required=True)

    # --- app ---
    app = sub.add_parser(
        "app", help="Run the Streamlit app (expects preprocessing already done)")
    app.add_argument(
        "--data-dir",
        default=None,
        help="Directory containing station_metadata/ and cache_parquet/. "
             "If omitted, uses TRAFFIC_VIZ_DATA_DIR or repo-local ./data or ~/traffic_viz-data.",
    )

    # --- preprocess ---
    prep = sub.add_parser(
        "preprocess", help="Build cache_parquet/ from raw 5-min data")
    prep.add_argument("--data-dir", default=None)
    prep.add_argument(
        "--pattern",
        default="d*_text_station_5min_*.txt.gz",
        help="Glob pattern for raw 5-min files inside 5min_data/ (default: %(default)s)",
    )
    prep.add_argument("--force", action="store_true",
                      help="Rebuild cache_parquet/ even if it exists.")

    # --- optional: doctor ---
    doc = sub.add_parser(
        "doctor", help="Check that data folders exist and look sane")
    doc.add_argument("--data-dir", default=None)

    args = p.parse_args()

    # Resolve data dir, and export it so app.py uses same directory.
    if args.data_dir:
        data_dir = Path(args.data_dir).expanduser().resolve()
        os.environ["TRAFFIC_VIZ_DATA_DIR"] = str(data_dir)
    else:
        data_dir = get_data_dir()
        os.environ["TRAFFIC_VIZ_DATA_DIR"] = str(data_dir)

    meta_dir = data_dir / "station_metadata"
    raw_dir = data_dir / "5min_data"
    cache_dir = data_dir / "cache_parquet"

    if args.cmd == "doctor":
        print(f"[INFO] DATA_DIR = {data_dir}")
        print(
            f"[INFO] station_metadata/: {'OK' if _has_station_metadata(meta_dir) else 'MISSING/EMPTY'} ({meta_dir})")
        print(
            f"[INFO] 5min_data/:        {'OK' if _has_raw_5min(raw_dir) else 'MISSING/EMPTY'} ({raw_dir})")
        print(
            f"[INFO] cache_parquet/:    {'OK' if _has_parquet_cache(cache_dir) else 'MISSING/EMPTY'} ({cache_dir})")
        if not _has_parquet_cache(cache_dir):
            print("\n[HINT] Run: traffic_viz preprocess --data-dir", data_dir)
        return

    if args.cmd == "preprocess":
        cache_dir.mkdir(parents=True, exist_ok=True)

        if _has_parquet_cache(cache_dir) and not args.force:
            print(
                f"[OK] cache_parquet already exists with parquet files: {cache_dir}")
            print("[HINT] Use --force to rebuild.")
            return

        if not _has_station_metadata(meta_dir):
            raise FileNotFoundError(
                f"station_metadata missing/empty at: {meta_dir}\n"
                "Expected files like: d07_text_meta_*.txt, d11_text_meta_*.txt, d12_text_meta_*.txt"
            )

        if not _has_raw_5min(raw_dir):
            raise FileNotFoundError(
                f"raw 5min_data missing/empty at: {raw_dir}\n"
                "Expected .txt.gz station 5-min files inside 5min_data/."
            )

        print(f"[INFO] Preprocessing raw 5-min files from: {raw_dir}")
        print(f"[INFO] Writing parquet cache to: {cache_dir}")
        preprocess_dir(raw_dir=raw_dir, out_parquet=cache_dir,
                       pattern=args.pattern)
        print("[OK] Preprocess complete.")
        return

    if args.cmd == "app":
        if not _has_parquet_cache(cache_dir):
            raise FileNotFoundError(
                f"cache_parquet missing/empty at: {cache_dir}\n"
                "Run preprocessing first:\n"
                f"  traffic_viz preprocess --data-dir {data_dir}\n"
                "Then run:\n"
                f"  traffic_viz app --data-dir {data_dir}"
            )
        run_streamlit_app()
        return


if __name__ == "__main__":
    main()
