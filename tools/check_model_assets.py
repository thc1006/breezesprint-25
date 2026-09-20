"""Online, CPU-only check of the real publisher metadata. Never downloads weights.

Run: JAX_PLATFORMS=cpu python tools/check_model_assets.py
A network failure is a failed check, never a silently skipped/pass result.
"""
from pathlib import Path
import argparse
import json
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from breezesprint25.assets import profile, fetch_snapshot
from breezesprint25.preflight import METADATA_FILES, validate_model_assets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=ROOT/'.cache/model-metadata')
    parser.add_argument('--report', type=Path, default=ROOT/'evidence/publisher-preflight.json')
    args = parser.parse_args()
    p = profile()
    report = dict(passed=False, model_id=p['model_id'], revision=p['revision'],
                  weights_loaded=False, tpu_inference_tested=False)
    try:
        files, identity = fetch_snapshot(p['preset'], args.cache_dir, METADATA_FILES)
        _, _, report = validate_model_assets(files, p)
        report['files_integrity'] = identity['files_integrity']
        report['files_sha256'] = identity['files_sha256']
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
        return 1
    finally:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
