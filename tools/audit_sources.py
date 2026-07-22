#!/usr/bin/env python3
"""Validate original and multilayer corpus invariants."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.corpus_data import AuditError, build_audit, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, help="Optional path for the JSON audit manifest.")
    parser.add_argument("--json", action="store_true", help="Print the complete manifest as JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        audit = build_audit()
    except (AuditError, OSError, ValueError) as exc:
        print(f"AUDIT FAILED: {exc}", file=sys.stderr)
        return 1

    manifest = audit.manifest()
    if args.manifest:
        write_json(args.manifest, manifest)
    if args.json:
        print(json.dumps(manifest, indent=2, sort_keys=True))
    else:
        totals = audit.totals
        print(
            "AUDIT PASSED: "
            f"{totals['documents']} documents, {totals['adus']} ADUs, "
            f"{totals['english_edus']} English EDUs, "
            f"{totals['split_adus']} split ADUs, "
            f"{totals['internal_boundaries']} internal boundaries, "
            f"{totals['sameunit_documents']} Same-Unit alternatives"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
