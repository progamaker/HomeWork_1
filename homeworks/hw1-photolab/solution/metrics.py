# -*- coding: utf-8 -*-
"""Числа для REPORT.md, которых нет в выводе photolab.py.

Эталон есть только у кадров курса, поэтому recall / precision маски и совпадение
выпрямленного кадра с эталоном считаются отдельно — «три строки NumPy», о которых
говорит sample_data/README.md.

    python metrics.py recall out/ck_mask.png --circle 190,560,58
    python metrics.py recall out/object_mask.png --gt data/object_gt.png
    python metrics.py match  out/front640.jpg lectures/L04-geometry/demo/poster-gt.png

recall  — какая доля эталонного объекта попала в маску (сколько объекта не потеряли);
precision — какая доля маски действительно объект (сколько фона нахватали).
"""
import argparse
import json
import sys

import cv2
import numpy as np


def read_mask(path):
    """Маску читаем как есть: всё ярче 127 — объект."""
    m = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise SystemExit("не удалось прочитать маску: %s" % path)
    return m > 127


def circle_gt(shape, cx, cy, r):
    """Эталон-круг: мяч на кадрах курса нарисован, поэтому его границу знаем точно."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def recall_precision(mask, gt):
    tp = int((mask & gt).sum())
    return (100.0 * tp / max(int(gt.sum()), 1),
            100.0 * tp / max(int(mask.sum()), 1))


def cmd_recall(a):
    mask = read_mask(a.mask)
    if a.circle:
        cx, cy, r = (int(v) for v in a.circle.split(","))
        gt = circle_gt(mask.shape, cx, cy, r)
        ref = "круг (%d, %d, r = %d)" % (cx, cy, r)
    else:
        gt = read_mask(a.gt)
        if gt.shape != mask.shape:
            raise SystemExit("размеры не совпали: маска %s, эталон %s" % (mask.shape, gt.shape))
        ref = a.gt
    rec, prec = recall_precision(mask, gt)
    print(json.dumps({"mask": a.mask, "gt": ref, "recall_pct": round(rec, 1), "precision_pct": round(prec, 1),
                      "mask_pct": round(100.0 * mask.mean(), 2), "gt_pct": round(100.0 * gt.mean(), 2)},
                     ensure_ascii=False))


def cmd_match(a):
    """Доля совпавших пикселей после бинаризации: так сверяют выпрямление с poster-gt.png."""
    got = cv2.imdecode(np.fromfile(a.image, np.uint8), cv2.IMREAD_GRAYSCALE)
    ref = cv2.imdecode(np.fromfile(a.reference, np.uint8), cv2.IMREAD_GRAYSCALE)
    if got is None or ref is None:
        raise SystemExit("не удалось прочитать один из кадров")
    if got.shape != ref.shape:
        raise SystemExit("размеры не совпали: %s против %s — выпрямляйте с --size %dx%d"
                         % (got.shape, ref.shape, ref.shape[1], ref.shape[0]))
    a_ink, b_ink = got < a.threshold, ref < a.threshold          # «краска» — тёмные пиксели
    print(json.dumps({"image": a.image, "reference": a.reference, "threshold": a.threshold,
                      "match_pct": round(100.0 * float((a_ink == b_ink).mean()), 1),
                      "ink_pct_image": round(100.0 * float(a_ink.mean()), 1),
                      "ink_pct_reference": round(100.0 * float(b_ink.mean()), 1)}, ensure_ascii=False))


def main(argv=None):
    p = argparse.ArgumentParser(prog="metrics", description="recall / precision маски и совпадение с эталоном")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("recall", help="recall и precision маски против эталона")
    s.add_argument("mask")
    g = s.add_mutually_exclusive_group(required=True)
    g.add_argument("--circle", help="x,y,r — эталон-круг (мяч кадров курса: 190,560,58)")
    g.add_argument("--gt", help="PNG эталонной маски")
    s.set_defaults(fn=cmd_recall)

    s = sub.add_parser("match", help="доля совпавших пикселей после бинаризации")
    s.add_argument("image"); s.add_argument("reference")
    s.add_argument("--threshold", type=int, default=128)
    s.set_defaults(fn=cmd_match)

    a = p.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
    sys.exit(main())
