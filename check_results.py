"""Sanity-check a full results.jsonl against the Parquet files it came from.

    python check_results.py --input finevideo/data --results results.jsonl

Checks: every row of every file has output, every video's seconds run
0..last with no gaps or duplicates, and lists videos with error lines.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--results", required=True)
    args = ap.parse_args()

    secs, errors, bad_lines = defaultdict(list), {}, 0
    with open(args.results) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                bad_lines += 1
                continue
            if "error" in r:
                errors[r["video"]] = r["error"]
            else:
                secs[r["video"]].append(r["second"])

    expected = []
    for p in sorted(Path(args.input).glob("*.parquet")):
        try:
            n = pq.ParquetFile(p).metadata.num_rows
        except Exception as e:
            print(f"cannot read {p.name}: {e}")
            continue
        expected += [f"{p.name}:{i}" for i in range(n)]

    missing = [v for v in expected if v not in secs and v not in errors]
    gaps = [v for v, s in secs.items() if sorted(s) != list(range(max(s) + 1))]
    total_secs = sum(len(s) for s in secs.values())

    print(f"expected videos: {len(expected)}   with seconds: {len(secs)}   "
          f"error lines: {len(errors)}   unparseable lines: {bad_lines}")
    print(f"total seconds: {total_secs}  (= {total_secs / 3600:.1f} hours of video)")
    print(f"videos with NO output at all: {len(missing)}  {missing[:5]}")
    print(f"videos with gaps/duplicate seconds: {len(gaps)}  {gaps[:5]}")
    for v, e in list(errors.items())[:10]:
        print(f"  error {v}: {e}")
    ok = not missing and not gaps and not bad_lines
    print("RESULT:", "OK" if ok else "PROBLEMS FOUND")


if __name__ == "__main__":
    main()
