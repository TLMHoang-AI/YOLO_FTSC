"""Merge verified frozen paper results with corrected Color/Edge, without inference.

Reference revision is explicit; source-hashed evaluation modules and metrics are
never rewritten. Existing recovery audit and cross-case artifact checks remain
mandatory before any mean/sample-standard-deviation table is emitted.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from utils import ftsc_deployment_levir_edge_color_recovery_v1 as recovery

REFERENCE_REVISIONS = (
    "predict512_native_rect_dynamic_trt_v1",
    "predict512_native_rect_dynamic_trt_source_identity_v2",
)


def require_complete(report):
    recovery.require(report.get("issues") == []
                     and report.get("valid_jobs") == report.get("expected_jobs") == 24,
                     "Corrected Color/Edge results incomplete or invalid")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--reference-revision", required=True, choices=REFERENCE_REVISIONS)
    options, rest = parser.parse_known_args(argv)
    args = recovery.parsed_options(rest)
    recovery.require(args.phase == "aggregate" and args.stage == "repair_all_precision",
                     "Merge requires aggregate/repair_all_precision")
    recovery.validate_recovery_manifest(recovery.MANIFEST)
    recovery.install_process_contract()
    # Explicit source-selection configuration, not relabeling stored records.
    # audit_old_frozen_three checks each record's actual revision against this
    # selection, as well as physical artifacts, source SHA, GT and pixel parity.
    recovery.ORIGINAL_PAPER_REVISION = options.reference_revision
    fresh = recovery.audit_recovery(args)
    require_complete(fresh)
    old, old_root = recovery.audit_old_frozen_three(args)
    recovery.require(len(old) == 36, "Expected all 36 verified reference records")
    fresh_rows = {(row["case"], row["backend"]): row for row in fresh["three_seed"]}
    recovery.require(len(fresh_rows) == 8 and all(r["n"] == 3 for r in fresh_rows.values()),
                     "Corrected Color/Edge three-seed summaries incomplete")
    flags = list(fresh["accuracy_review_flags"])
    historical = list(fresh["historical_reference_review_flags"])
    keys = recovery.SCORE_KEYS
    # Also retain numerical review findings from the three restored cases.
    for case in recovery.THREE_VERIFIED:
        for seed in (42, 43, 44):
            for backend in recovery.BACKENDS:
                if backend == "pt32":
                    continue
                reference = "pt16" if backend == "trt16" else "pt32"
                row, ref = old[(case, seed, backend)], old[(case, seed, reference)]
                deltas = {key: 100*(row["metrics"][key]-ref["metrics"][key]) for key in keys}
                limit = 0.5 if backend == "trt32" else 1.0
                if any(abs(value) > limit for value in deltas.values()):
                    flags.append({"case": case, "seed": seed, "backend": backend,
                                  "vs": reference, "metric_deltas_pp": deltas,
                                  "review_limit_pp": limit, "source_batch": "existing_reference"})
    status = "COMPLETE_NEEDS_ACCURACY_REVIEW" if flags or historical else "PASS_COMPLETE"
    rows = []
    for case in recovery.FIVE_CASES:
        for backend in recovery.BACKENDS:
            existing = case in recovery.THREE_VERIFIED
            row = (recovery.aggregate_three_seed(case, backend, old) if existing
                   else dict(fresh_rows[(case, backend)]))
            rows.append({
                **row, "audit_status": status,
                "source_batch": "existing_sha_audited" if existing else "new_corrected_paper_checkpoint",
                "source_revision": options.reference_revision if existing else recovery.REVISION,
                "backend_accuracy_review_flags": json.dumps([
                    f for f in flags if f["case"] == case and backend in (f["backend"], f["vs"])
                ], sort_keys=True),
                "historical_reference_review_flags": json.dumps([
                    f for f in historical if f["case"] == case and backend == f["backend"]
                ], sort_keys=True),
            })
    recovery.require(len(rows) == 20 and all(row["n"] == 3 for row in rows),
                     "Final five-case matrix incomplete")
    destination = args.paper_root/"LEVIR-Ship"/"_recovery_reports"/options.reference_revision
    destination.mkdir(parents=True, exist_ok=True)
    target = destination/"levir_paper_five_accuracy_review_labeled_v2.csv"
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=sorted(set().union(*(row.keys() for row in rows))),
                            lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    value = buffer.getvalue()
    if target.exists():
        recovery.require(target.read_text() == value, "Existing merged CSV differs; refuse overwrite")
    else:
        target.write_text(value, encoding="utf-8")
    summary = {
        "status": status, "dataset": "LEVIR-Ship", "rows": 20, "valid_jobs": 60,
        "scope": "5 paper configurations x 3 seeds x 4 backends",
        "reference_jobs": 36, "corrected_jobs": 24,
        "reference_revision": options.reference_revision, "reference_root": str(old_root),
        "recovery_revision": recovery.REVISION, "new_recovery_audit": fresh,
        "accuracy_review_flags": flags, "historical_reference_review_flags": historical,
        "aggregate_rows": rows, "csv": str(target),
        "merge_helper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "note": "Existing metrics/checkpoints unchanged. Review flags are retained, not waived.",
    }
    recovery.safe_write(target.with_suffix(".json"), summary)
    print(json.dumps({"status": status, "rows": 20, "valid_jobs": 60,
                      "accuracy_review_flags": len(flags), "historical_reference_review_flags": len(historical),
                      "csv": str(target), "json": str(target.with_suffix('.json'))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
