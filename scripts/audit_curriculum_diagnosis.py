#!/usr/bin/env python3
"""Compare saved human labels to the ordered Gemma batch; emit an unapproved report."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from diffusionopsd.curriculum.audit_review import compare_review
from diffusionopsd.curriculum.proposer import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    labels = json.loads(args.labels.read_text())
    predictions = json.loads(args.predictions.read_text())
    report, disagreements = compare_review(manifest, labels['rows'], predictions)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / 'agreement_report.json', report)
    atomic_json(args.output_dir / 'disagreements.json', disagreements)
    with (args.output_dir / 'agreement_summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['requirement', 'count', 'agreement', 'false_pass', 'false_fail',
                         'overconfident_on_unknown', 'unknown_on_decidable', 'missing_human_classes'])
        for requirement, result in report['requirements'].items():
            writer.writerow([requirement, result['count'], result['agreement_rate'], result.get('false_pass', 0),
                result.get('false_fail', 0), result.get('overconfident_on_unknown', 0),
                result.get('unknown_on_decidable', 0), ','.join(result['missing_human_classes'])])
    print(f'Compared {report["clip_count"]} clips; {len(disagreements)} disagreements. Report remains unapproved.')


if __name__ == '__main__':
    main()
