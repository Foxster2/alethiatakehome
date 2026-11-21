"""Score a results.jsonl against a reference, the way the brief describes.

    python score.py --ref reference_2videos.jsonl --pred results.jsonl

Checks (from the brief, section 5):
  * objects F1 >= 0.95       (pooled over every (video, second, object))
  * people exact  >= 85%     of reference seconds
  * people within 1 >= 95%   of reference seconds
"""

import argparse
import json
from collections import Counter

NAMES = ["person", "car", "dog", "laptop", "cell phone"]
THRESH = {"f1": 0.95, "exact": 0.85, "within1": 0.95}


def load(path):
    """-> ({(video, second): (set_of_objects, people)}, {video: error})"""
    secs, errors = {}, {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if "error" in r:
                errors[r["video"]] = r["error"]
            else:
                secs[(r["video"], r["second"])] = (set(r["objects"]), r["people"])
    return secs, errors


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()

    ref, ref_err = load(args.ref)
    pred, pred_err = load(args.pred)

    tp, fp, fn = Counter(), Counter(), Counter()
    exact = within1 = 0
    missing = 0

    for key, (ref_objs, ref_people) in ref.items():
        if key in pred:
            pred_objs, pred_people = pred[key]
        else:                      # second missing in our output = all wrong
            missing += 1
            pred_objs, pred_people = set(), None
        for name in NAMES:
            if name in ref_objs and name in pred_objs:
                tp[name] += 1
            elif name in pred_objs:
                fp[name] += 1
            elif name in ref_objs:
                fn[name] += 1
        if pred_people is not None:
            exact += pred_people == ref_people
            within1 += abs(pred_people - ref_people) <= 1

    extra = [k for k in pred if k not in ref]   # seconds the reference lacks
    for key in extra:
        for name in pred[key][0]:
            fp[name] += 1

    n = len(ref)
    F1, P, R = f1(sum(tp.values()), sum(fp.values()), sum(fn.values()))
    scores = {"f1": F1, "exact": exact / n if n else 0, "within1": within1 / n if n else 0}

    print(f"reference seconds: {n}   missing in pred: {missing}   extra in pred: {len(extra)}")
    if ref_err or pred_err:
        print(f"error lines -> ref: {len(ref_err)}  pred: {len(pred_err)}")
    print("\nper object        TP    FP    FN     F1")
    for name in NAMES:
        if tp[name] + fp[name] + fn[name]:
            print(f"  {name:<12} {tp[name]:5d} {fp[name]:5d} {fn[name]:5d}  {f1(tp[name], fp[name], fn[name])[0]:.3f}")
    print(f"\nprecision {P:.4f}   recall {R:.4f}")
    ok_all = True
    for k, label in [("f1", "objects F1"), ("exact", "people exact"), ("within1", "people within 1")]:
        ok = scores[k] >= THRESH[k]
        ok_all &= ok
        print(f"{label:<17} {scores[k]:.4f}  (need >= {THRESH[k]})  {'PASS' if ok else 'FAIL'}")
    print("\nOVERALL:", "PASS" if ok_all else "FAIL")


if __name__ == "__main__":
    main()
