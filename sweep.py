"""Choose the frame stride with evidence instead of guesswork.

Step 1 - log (GPU):  every frame of a random window of many videos goes
through YOLO twice (FP32 = the graders' reference, FP16 = what we submit).
Both per-frame results are saved, one JSON line per video.

    python sweep.py log --files finevideo/data/train-0000[2-6]-*.parquet \
        --videos 100 --window 90 --minutes 55 --out sweep_log.jsonl

Step 2 - replay (CPU, seconds): rebuild exactly what `run.py --stride k`
would output for every k, score each against the FP32 every-frame
reference, and bootstrap over videos to get a "worst plausible" score.

    python sweep.py replay --log sweep_log.jsonl

The replay uses the same sampling rule as run.py: inside every second,
frames j = 0, k, 2k, ... (j counted from the start of that second).
"""

import argparse
import io
import json
import math
import random
import sys
import time
from pathlib import Path

import av
import pyarrow.parquet as pq

import run  # reuse frame_time, upright, summarise rules, class ids

NAMES = run.NAME_ORDER                      # person, car, dog, laptop, cell phone
BIT = {name: 1 << i for i, name in enumerate(NAMES)}
THRESH = {"f1": 0.95, "exact": 0.85, "within1": 0.95}


# ====================================================================== log
def detect_masks(model, images, half):
    """YOLO on a batch -> list of (object_bitmask, people_count)."""
    precision = {"quantize": 16} if half else {}      # see run.Indexer.detect
    results = model.predict(images, imgsz=run.IMGSZ, conf=run.CONF,
                            classes=list(run.TARGETS), verbose=False, **precision)
    out = []
    for r in results:
        ids = r.boxes.cls.int().tolist()
        mask = 0
        for i in ids:
            mask |= BIT[run.TARGETS[i]]
        out.append((mask, ids.count(run.PERSON_ID)))
    return out


def read_row(path, row):
    """Read one row's mp4 bytes without loading the whole file."""
    pf = pq.ParquetFile(path)
    start = 0
    for g in range(pf.metadata.num_row_groups):
        n = pf.metadata.row_group(g).num_rows
        if row < start + n:
            col = pf.read_row_group(g, columns=["mp4"]).column("mp4")
            return col[row - start].as_py()
        start += n
    raise IndexError(f"row {row} not in {path}")


def pick_videos(files, n_videos, window, seed):
    rng = random.Random(seed)
    cands = []
    for f in files:
        try:
            meta = pq.read_table(f, columns=["json"]).column("json").to_pylist()
        except Exception as e:
            print(f"skip {f}: {e}", file=sys.stderr)
            continue
        for row, m in enumerate(meta):
            dur = (m or {}).get("duration_seconds") if isinstance(m, dict) else None
            cands.append((f, row, dur))
    rng.shuffle(cands)
    picked = []
    for f, row, dur in cands[:n_videos]:
        if dur and dur > window:
            ws = rng.randint(0, int(dur) - window)
        else:
            ws = 0
        picked.append((f, row, ws))
    return picked


def log_video(models, data, ws, we, batch):
    """Return [[sec, j, m32, p32, m16, p16], ...] for seconds ws <= sec < we."""
    m32, m16 = models
    rows, pending = [], []          # pending: (sec, j, image)

    def flush():
        if pending:
            imgs = [p[2] for p in pending]
            d32 = detect_masks(m32, imgs, False)
            d16 = detect_masks(m16, imgs, True)
            for (sec, j, _), a, b in zip(pending, d32, d16):
                rows.append([sec, j, a[0], a[1], b[0], b[1]])
            pending.clear()

    t0, cur_sec, j = None, None, 0
    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for i, frame in enumerate(container.decode(stream)):
            t = run.frame_time(frame, stream, i)
            if t0 is None:
                t0 = t
            sec = max(0, int(math.floor(t - t0)))
            if sec != cur_sec:
                cur_sec, j = sec, 0
            jj = j
            j += 1
            if sec < ws:
                continue                     # decode only, before the window
            if sec >= we:
                break                        # past the window: stop decoding
            img = run.upright(frame.to_ndarray(format="bgr24"), frame.rotation)
            pending.append((sec, jj, img))
            if len(pending) >= batch:
                flush()
        flush()
    return rows


def cmd_log(args):
    from ultralytics import YOLO
    files = [Path(f) for f in args.files]
    picked = pick_videos(files, args.videos, args.window, args.seed)
    done = set()
    if Path(args.out).exists():                      # resume
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["video"])
                except Exception:
                    pass
    models = (YOLO(args.weights), YOLO(args.weights))
    deadline = time.time() + args.minutes * 60
    start = time.time()
    n_ok = n_frames = 0
    with open(args.out, "a") as out:
        for k, (f, row, ws) in enumerate(picked):
            vid = f"{f.name}:{row}"
            if vid in done:
                continue
            if time.time() > deadline:
                print("time box reached, stopping", file=sys.stderr)
                break
            try:
                rows = log_video(models, read_row(f, row), ws, ws + args.window, args.batch)
                if not rows:
                    raise RuntimeError("no frames in window")
                out.write(json.dumps({"video": vid, "window": [ws, ws + args.window],
                                      "frames": rows}) + "\n")
                out.flush()
                n_ok += 1
                n_frames += len(rows)
            except Exception as e:
                print(f"  {vid}: {type(e).__name__}: {e}", file=sys.stderr)
            el = time.time() - start
            print(f"[{el:6.0f}s] {k + 1}/{len(picked)} {vid}  logged={n_ok}  "
                  f"frames={n_frames} ({n_frames / max(el, 1e-9):.0f}/s)", file=sys.stderr)


# =================================================================== replay
def second_result(frames, prec):
    """frames: list of (j, m32, p32, m16, p16) -> (mask, people)."""
    n = len(frames)
    mi, pi = (1, 2) if prec == "fp32" else (3, 4)
    mask = 0
    for name, bit in BIT.items():
        if 2 * sum(1 for fr in frames if fr[mi] & bit) >= n:
            mask |= bit
    return mask, run.lower_median([fr[pi] for fr in frames])


def per_video_stats(by_sec, stride, prec):
    """Per-video counts against the FP32 every-frame reference."""
    s = {"tp": 0, "fp": 0, "fn": 0, "exact": 0, "within1": 0, "n": 0,
         "sent": 0, "total": 0}
    for name in NAMES:
        s[f"tp_{name}"] = s[f"fp_{name}"] = s[f"fn_{name}"] = 0
    for sec, frames in by_sec.items():
        ref_mask, ref_people = second_result(frames, "fp32")
        sample = [fr for fr in frames if fr[0] % stride == 0]
        mask, people = second_result(sample, prec)
        for name, bit in BIT.items():
            r, p = bool(ref_mask & bit), bool(mask & bit)
            key = "tp" if (r and p) else "fp" if p else "fn" if r else None
            if key:
                s[key] += 1
                s[f"{key}_{name}"] += 1
        s["exact"] += people == ref_people
        s["within1"] += abs(people - ref_people) <= 1
        s["n"] += 1
        s["sent"] += len(sample)
        s["total"] += len(frames)
    return s


def scores(stats_list):
    tp = sum(s["tp"] for s in stats_list)
    fp = sum(s["fp"] for s in stats_list)
    fn = sum(s["fn"] for s in stats_list)
    n = sum(s["n"] for s in stats_list)
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0
    return {"f1": f1,
            "exact": sum(s["exact"] for s in stats_list) / n,
            "within1": sum(s["within1"] for s in stats_list) / n}


def results_lines(vid, by_sec, stride, prec):
    for sec in sorted(by_sec):
        sample = [fr for fr in by_sec[sec] if fr[0] % stride == 0]
        mask, people = second_result(sample, prec)
        yield {"video": vid, "second": sec,
               "objects": [n for n in NAMES if mask & BIT[n]], "people": people}


def cmd_replay(args):
    videos = []
    with open(args.log) as f:
        for line in f:
            r = json.loads(line)
            by_sec = {}
            for sec, j, a, b, c, d in r["frames"]:
                by_sec.setdefault(sec, []).append((j, a, b, c, d))
            videos.append((r["video"], by_sec))
    if not videos:
        sys.exit("empty log")

    if args.dump:                      # write run.py-format output for one config
        with open(args.dump, "w") as out:
            for vid, by_sec in videos:
                for obj in results_lines(vid, by_sec, args.dump_stride, args.dump_prec):
                    out.write(json.dumps(obj) + "\n")
        print(f"wrote {args.dump}")
        return

    n_secs = sum(len(b) for _, b in videos)
    n_frames = sum(len(fr) for _, b in videos for fr in b.values())
    print(f"videos: {len(videos)}   seconds: {n_secs}   frames: {n_frames}   "
          f"bootstrap: {args.boot} resamples over videos (5th percentile shown)\n")

    rng = random.Random(args.seed)
    boots = [[rng.randrange(len(videos)) for _ in videos] for _ in range(args.boot)]

    configs = [("fp32", 1)] + [("fp16", k) for k in range(1, args.max_stride + 1)]
    print(f"{'config':<12}{'YOLO frames':>12}  {'F1':>6} {'(p5)':>7}  "
          f"{'exact':>6} {'(p5)':>7}  {'within1':>7} {'(p5)':>7}   verdict on p5")
    table = {}
    for prec, k in configs:
        per_vid = [per_video_stats(b, k, prec) for _, b in videos]
        point = scores(per_vid)
        bs = {m: [] for m in THRESH}
        for idx in boots:
            sc = scores([per_vid[i] for i in idx])
            for m in THRESH:
                bs[m].append(sc[m])
        p5 = {m: sorted(v)[int(0.05 * len(v))] for m, v in bs.items()}
        sent = sum(s["sent"] for s in per_vid) / max(1, sum(s["total"] for s in per_vid))
        ok = all(p5[m] >= THRESH[m] for m in THRESH)
        margin = min(p5[m] - THRESH[m] for m in THRESH)
        table[(prec, k)] = per_vid
        print(f"{prec + ' s' + str(k):<12}{100 * sent:>11.1f}%  "
              f"{point['f1']:.4f} {p5['f1']:.4f}  {point['exact']:.4f} {p5['exact']:.4f}  "
              f"{point['within1']:.4f}  {p5['within1']:.4f}   "
              f"{'PASS' if ok else 'FAIL'} (worst margin {margin:+.3f})")

    print("\nper-object F1 (FP16), pooled; counts are reference-present seconds")
    print(f"{'stride':<8}" + "".join(f"{n:>13}" for n in NAMES))
    for k in range(1, args.max_stride + 1):
        per_vid = table[("fp16", k)]
        cells = []
        for name in NAMES:
            tp = sum(s[f"tp_{name}"] for s in per_vid)
            fp = sum(s[f"fp_{name}"] for s in per_vid)
            fn = sum(s[f"fn_{name}"] for s in per_vid)
            cells.append(f"{(2 * tp / (2 * tp + fp + fn)) if tp + fp + fn else float('nan'):.3f}"
                         f" ({tp + fn})")
        print(f"{k:<8}" + "".join(f"{c:>13}" for c in cells))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("log")
    a.add_argument("--files", nargs="+", required=True)
    a.add_argument("--videos", type=int, default=100)
    a.add_argument("--window", type=int, default=90, help="seconds per video")
    a.add_argument("--minutes", type=float, default=55, help="time box")
    a.add_argument("--out", default="sweep_log.jsonl")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--batch", type=int, default=32)
    a.add_argument("--weights", default="yolo11l.pt")
    b = sub.add_parser("replay")
    b.add_argument("--log", default="sweep_log.jsonl")
    b.add_argument("--max-stride", type=int, default=10)
    b.add_argument("--boot", type=int, default=1000)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--dump", help="(test) write run.py-format output to this file")
    b.add_argument("--dump-stride", type=int, default=1)
    b.add_argument("--dump-prec", default="fp16")
    args = ap.parse_args()
    cmd_log(args) if args.cmd == "log" else cmd_replay(args)


if __name__ == "__main__":
    main()
