"""Build a small, varied dev set from one FineVideo Parquet file.

Picks short videos from as many different content categories as possible
(plus portrait videos, to exercise rotation), so that a full every-frame
reference on them stays cheap.

    python make_devset.py --src finevideo/data/train-00001-of-01357.parquet \
                          --out devset --budget 900
"""

import argparse
from pathlib import Path

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="source .parquet file")
    ap.add_argument("--out", required=True, help="output folder")
    ap.add_argument("--max-dur", type=float, default=150, help="longest video to pick (s)")
    ap.add_argument("--budget", type=float, default=900, help="total seconds of video")
    args = ap.parse_args()

    meta = pq.read_table(args.src, columns=["json"]).column("json").to_pylist()
    rows = []
    for i, m in enumerate(meta):
        m = m if isinstance(m, dict) else {}
        res = str(m.get("resolution") or "0x0")
        try:
            w, h = (int(x) for x in res.lower().split("x"))
        except ValueError:
            w, h = 0, 0
        rows.append({"row": i, "dur": float(m.get("duration_seconds") or 1e9),
                     "cat": m.get("content_parent_category") or "unknown",
                     "res": res, "portrait": h > w})

    short = sorted((r for r in rows if r["dur"] <= args.max_dur), key=lambda r: r["dur"])
    picked, cats, total = [], set(), 0.0

    def take(r):
        nonlocal total
        if r not in picked and total + r["dur"] <= args.budget:
            picked.append(r)
            cats.add(r["cat"])
            total += r["dur"]

    for r in [r for r in short if r["portrait"]][:2]:   # rotation / portrait coverage
        take(r)
    for r in short:                                      # one per category first
        if r["cat"] not in cats:
            take(r)
    for r in short:                                      # then fill the budget
        take(r)

    picked.sort(key=lambda r: r["row"])
    table = pq.read_table(args.src).take([r["row"] for r in picked])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dst = out / ("devset-" + Path(args.src).name)
    pq.write_table(table, dst)

    print(f"wrote {dst}: {len(picked)} videos, {total:.0f} s of video")
    for k, r in enumerate(picked):
        print(f"  :{k:<3} (source row {r['row']:>2})  {r['dur']:5.0f}s  {r['res']:>9}  {r['cat']}")


if __name__ == "__main__":
    main()
