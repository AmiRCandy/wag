"""Check the archived manuscript run by file content (SHA-256), independent of absolute paths.

The run manifests record files with the absolute paths of the machine that produced them.
This script maps those paths to this repository and compares content hashes. Files that
are not distributed (fitted models, intermediate task records, the font) are reported as
not included rather than as failures.
"""

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "runs" / "run_20261006T134517Z_6015c98f"
ORIGINAL_ROOT = "/mnt/disk3/saberi/co2-wag-pinn-saber/co2-wag-pinn/"
# Files moved when the repository was prepared for publication.
RELOCATED = {
    "WAG_simulation_dataset_B_train.csv": "data/WAG_simulation_dataset_B_train.csv",
    "WAG_simulation_dataset_A_test.csv": "data/WAG_simulation_dataset_A_test.csv",
}


def recorded_fingerprints(node, found):
    if isinstance(node, dict):
        path = node.get("path")
        if isinstance(path, str) and path.startswith(ORIGINAL_ROOT) and "sha256" in node:
            found.setdefault(path[len(ORIGINAL_ROOT):], set()).add(node["sha256"])
            return
        for value in node.values():
            recorded_fingerprints(value, found)
    elif isinstance(node, list):
        for value in node:
            recorded_fingerprints(value, found)


def main():
    found = {}
    for manifest in sorted(RUN.rglob("*.json")):
        recorded_fingerprints(json.loads(manifest.read_text(encoding="utf-8")), found)
    matched, changed, absent = [], [], []
    for relative, hashes in sorted(found.items()):
        path = ROOT / RELOCATED.get(relative, relative)
        if not path.is_file():
            absent.append(relative)
        elif hashlib.sha256(path.read_bytes()).hexdigest() in hashes:
            matched.append(relative)
        else:
            changed.append(relative)
    print(f"matching content: {len(matched)}")
    print(f"not included in this repository: {len(absent)}")
    print(f"different from the recorded version: {len(changed)}")
    for relative in changed:
        print(f"  {relative}")
    data_or_results = [r for r in changed if not r.startswith(("wag/", "figures"))]
    return 1 if data_or_results else 0


if __name__ == "__main__":
    sys.exit(main())
