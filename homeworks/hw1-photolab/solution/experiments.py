# -*- coding: utf-8 -*-
"""Два опыта из REPORT.md, которые не сводятся к одному запуску photolab.py.

    python experiments.py rgb-vs-hsv --day L3/ball-day.jpg --lamp L3/ball-lamp.jpg
    python experiments.py hue-shift L3/ball-day.jpg --shift 27 -o out/ball-red.png

`rgb-vs-hsv` — ответ на вопрос 2.1 отчёта: границы снимаются с одного прямоугольника
на мяче и применяются к двум кадрам одной сцены (день и лампа) по HSV и прямо по B, G, R.
Эталон известен точно (мяч нарисован), поэтому recall и precision считаются здесь же.

`hue-shift` — кадр для раздела 3 «где сломалось». Красного предмета ни на одном кадре
курса нет, а нужен именно он: у красного оттенки лежат по обе стороны разрыва 0/179,
и `hsv_box_from` на них ломается. Сдвиг оттенка на +27 ставит мяч `ball-day.jpg` ровно
на разрыв — сцена, свет и эталонный круг остаются теми же, меняется только оттенок.
Дальше кадр обрабатывается обычными командами photolab.py и metrics.py.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import photolab as pl  # noqa: E402


def circle_gt(shape, cx, cy, r):
    """Эталон-круг: мяч на кадрах курса нарисован, поэтому его границу знаем точно."""
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2


def recall_precision(mask, gt):
    tp = int((mask & gt).sum())
    return (round(100.0 * tp / max(int(gt.sum()), 1), 1),
            round(100.0 * tp / max(int(mask.sum()), 1), 1))


def _triple(s):
    return tuple(int(v) for v in s.split(","))


def bgr_box_from(bgr_pixels, margin=10, p=(1, 99)):
    """Тот же рецепт, что `hsv_box_from`, но прямо по B, G, R — для сравнения в вопросе 2.1."""
    x = bgr_pixels.reshape(-1, 3)
    lo = np.clip(np.floor(np.percentile(x, p[0], axis=0) - margin), 0, 255)
    hi = np.clip(np.ceil(np.percentile(x, p[1], axis=0) + margin), 0, 255)
    return tuple(int(v) for v in lo), tuple(int(v) for v in hi)


def _score(img, lo, hi, gt, space):
    """Маска по границам двумя способами: сырой inRange и полный путь команды (морфология + компонента)."""
    src = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) if space == "hsv" else img
    raw = cv2.inRange(src, np.array(lo, np.uint8), np.array(hi, np.uint8)) > 0
    rec, prec = recall_precision(raw, gt)
    out = {"raw": {"recall_pct": rec, "precision_pct": prec, "mask_pct": round(100.0 * float(raw.mean()), 2)}}
    if space == "hsv":
        found = pl.largest_component(pl.color_mask(img, lo, hi))
    else:                                                   # у B, G, R разрыва оттенка нет: та же морфология напрямую
        m = cv2.morphologyEx(raw.astype(np.uint8) * 255, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        found = pl.largest_component(cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))))
    if found is None:
        out["pipeline"] = None                              # компоненты ≥ 200 px нет — команда сказала бы «объект не найден»
    else:
        mask, st = found
        rec, prec = recall_precision(mask > 0, gt)
        out["pipeline"] = {"recall_pct": rec, "precision_pct": prec,
                           "mask_pct": round(100.0 * float((mask > 0).mean()), 2), "bbox": st["bbox"]}
    return out


def cmd_rgb_vs_hsv(a):
    x, y, w, h = a.sample
    cx, cy, r = a.circle
    day, lamp = pl.imread(a.day), pl.imread(a.lamp)
    gt_day, gt_lamp = circle_gt(day.shape[:2], cx, cy, r), circle_gt(lamp.shape[:2], cx, cy, r)
    sample = day[y:y + h, x:x + w]

    hsv_lo, hsv_hi = pl.hsv_box_from(sample.reshape(-1, 3))
    bgr_lo, bgr_hi = bgr_box_from(sample)
    lamp_lo, lamp_hi = pl.hsv_box_from(lamp[y:y + h, x:x + w].reshape(-1, 3))   # насколько уехали границы под лампой

    rep = {"experiment": "rgb-vs-hsv", "sample_xywh": [x, y, w, h], "gt_circle": [cx, cy, r],
           "gt_pct": round(100.0 * float(gt_day.mean()), 2),
           "box_from_day": {"hsv": [list(hsv_lo), list(hsv_hi)], "bgr": [list(bgr_lo), list(bgr_hi)]},
           "box_from_lamp_hsv": [list(lamp_lo), list(lamp_hi)],
           "hsv": {"day": _score(day, hsv_lo, hsv_hi, gt_day, "hsv"),
                   "lamp_same_box": _score(lamp, hsv_lo, hsv_hi, gt_lamp, "hsv")},
           "bgr": {"day": _score(day, bgr_lo, bgr_hi, gt_day, "bgr"),
                   "lamp_same_box": _score(lamp, bgr_lo, bgr_hi, gt_lamp, "bgr")}}
    print(json.dumps(rep, ensure_ascii=False))


def cmd_hue_shift(a):
    cx, cy, r = a.circle
    img = pl.imread(a.image)
    gt = circle_gt(img.shape[:2], cx, cy, r)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    before = hsv[gt][:, 0]
    hsv[..., 0] = (hsv[..., 0].astype(np.int16) + int(a.shift)) % 180      # оттенок — величина круговая, поэтому mod 180
    out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    pl.imwrite(a.out, out)                                  # PNG: JPEG переквантовал бы оттенок и числа поехали бы

    after = cv2.cvtColor(pl.imread(a.out), cv2.COLOR_BGR2HSV)[gt][:, 0]
    hue = lambda v: {"p1": int(np.percentile(v, 1)), "p99": int(np.percentile(v, 99)),   # noqa: E731
                     "below_5_pct": round(100.0 * float((v < 5).mean()), 1),
                     "above_174_pct": round(100.0 * float((v > 174).mean()), 1)}
    print(json.dumps({"experiment": "hue-shift", "shift": int(a.shift), "gt_circle": [cx, cy, r],
                      "object_hue_before": hue(before), "object_hue_after": hue(after),
                      "hsv_box_from_sample": [list(v) for v in pl.hsv_box_from(
                          pl.imread(a.out)[a.sample[1]:a.sample[1] + a.sample[3],
                                            a.sample[0]:a.sample[0] + a.sample[2]].reshape(-1, 3))],
                      "out": a.out}, ensure_ascii=False))


def cmd_lecture_check(a):
    """Воспроизводит числа лекций на том же кадре — сверка решения с материалом курса.

    L3, демо 1.4 (стр. 30): тень создаётся искусственно — правые 40 % серого кадра гасятся
    до 0.15, «тень» = правые 300 столбцов, «светлая часть» = левые 60 %. Это не то же самое,
    что считает `local_contrast_report` в photolab.py (канал L пространства Lab, границы по
    20-му и 80-му процентилям), поэтому числа двух измерений несравнимы между собой.
    L4, стр. 19 и демо 2.4 (стр. 67): потеря углов без расширения холста и цена алиасинга.
    """
    img = pl.imread(a.image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    x, x0 = np.arange(w, dtype=np.float32), 0.6 * w                 # профиль света со слайда
    light = np.where(x < x0, 1.0, 1 - 0.85 * (x - x0) / (w - 1 - x0))
    uneven = np.uint8(gray * light.astype(np.float32))
    shadow = (slice(None), slice(w - 300, None))
    variants = (("вход", uneven), ("equalizeHist", cv2.equalizeHist(uneven)),
                ("CLAHE 2.0", cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(uneven)))
    demo14 = {n: {"shadow_std": round(float(im[shadow].std()), 1),
                  "light_mean": round(float(im[:, :int(x0)].mean()))} for n, im in variants}

    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 30, 1.0)
    kept = cv2.warpAffine(np.full((h, w), 255, np.uint8), M, (w, h)) > 127   # прежний dsize режет углы
    rot = pl.rotate_keep_all(img, 30)

    ref = pl.lanczos_reference(img, 0.25)
    psnr15 = pl.psnr(ref, cv2.resize(cv2.GaussianBlur(img, (0, 0), 1.5), (480, 320), interpolation=cv2.INTER_LINEAR))
    print(json.dumps({
        "experiment": "lecture-check", "image": a.image,
        "L3_демо_1.4": {"измерено": demo14,
                        "в_лекции": {"вход": {"shadow_std": 27.0, "light_mean": 116},
                                     "equalizeHist": {"shadow_std": 43.0, "light_mean": 139},
                                     "CLAHE 2.0": {"shadow_std": 44.9, "light_mean": 129}}},
        "L4_стр_19": {"холст_без_потерь": [rot.shape[1], rot.shape[0]], "в_лекции": [2302, 2068],
                      "потеряно_без_холста_pct": round(100 - 100 * float(kept.mean()), 1), "в_лекции_pct": 18},
        "L4_демо_2.4": {"naive": round(pl.psnr(ref, pl.resize_aa(img, 0.25, "naive")), 1),
                        "area": round(pl.psnr(ref, pl.resize_aa(img, 0.25, "area")), 1),
                        "blur_сигма_1.5_как_в_лекции": round(psnr15, 1),
                        "blur_шаблона_сигма_0.4_делить_scale": round(pl.psnr(ref, pl.resize_aa(img, 0.25, "blur")), 1),
                        "в_лекции": {"naive": 27.9, "area": 37.2, "blur": 36.7}}}, ensure_ascii=False))


def cmd_wb_reference(a):
    """Что баланс белого делает с *белым* в кадре — раздел 3 отчёта.

    Средние каналов по всему кадру grey-world выравнивает по построению, поэтому по ним
    метод всегда «успешен». Честная проверка — опорная поверхность, про которую известно,
    что она серая: светлые ненасыщенные пиксели исходника (стол, стена, лист).
    """
    img = pl.imread(a.image)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    ref = (hsv[..., 1] < a.max_sat) & (hsv[..., 2] >= np.percentile(hsv[..., 2], a.bright_pct))
    colored = (hsv[..., 1] > a.max_sat * 2) & (hsv[..., 2] > 80)     # крупная насыщенная область — враг grey-world
    if ref.sum() < 500:
        raise SystemExit("опорная серая поверхность не найдена: подберите --max-sat / --bright-pct")
    spread = lambda v: round(float(v.max() - v.min()), 1)            # noqa: E731
    rep = {"experiment": "wb-reference", "image": a.image,
           "reference_pct": round(100.0 * float(ref.mean()), 2),
           "colored_pct": round(100.0 * float(colored.mean()), 1), "methods": {}}
    m0 = img[ref].mean(0)
    rep["methods"]["исходник"] = {"reference_bgr": [round(float(v)) for v in m0], "spread": spread(m0)}
    for method in ("grey-world", "white-patch"):
        out, gains = pl.white_balance(img, method)
        m = out[ref].mean(0)
        rep["methods"][method] = {"gains_bgr": [round(float(g), 3) for g in gains],
                                  "reference_bgr": [round(float(v)) for v in m], "spread": spread(m),
                                  "frame_spread": spread(out.reshape(-1, 3).mean(0)),
                                  "blue_clipped_pct": round(100.0 * float((out[ref][:, 0] == 255).mean()), 1)}
    print(json.dumps(rep, ensure_ascii=False))


def aspect_from_quad(corners, shape):
    """W / H прямоугольника по его перспективному снимку (Zhang & He, «Whiteboard Scanning», 2003).

    Две точки схода задают фокусное расстояние, фокусное — метрические пропорции.
    Предположения: главная точка в центре кадра, пиксели квадратные. Возвращает
    (отношение, фокусное в px) или (None, None), если вид почти фронтальный:
    тогда точки схода уходят в бесконечность и задача вырождается.
    """
    c = pl.order_corners(corners)                                   # tl, tr, br, bl
    m1, m2, m4, m3 = (np.array([p[0], p[1], 1.0]) for p in c)       # m3 — bl, m4 — br
    u0, v0 = shape[1] / 2.0, shape[0] / 2.0
    k2 = np.dot(np.cross(m1, m4), m3) / np.dot(np.cross(m2, m4), m3)
    k3 = np.dot(np.cross(m1, m4), m2) / np.dot(np.cross(m3, m4), m2)
    n2, n3 = k2 * m2 - m1, k3 * m3 - m1
    den = n2[2] * n3[2]
    if abs(den) < 1e-12:
        return None, None
    f2 = -((n2[0] - n2[2] * u0) * (n3[0] - n3[2] * u0)
           + (n2[1] - n2[2] * v0) * (n3[1] - n3[2] * v0)) / den
    if f2 <= 0:
        return None, None
    W = np.array([[1, 0, -u0], [0, 1, -v0], [-u0, -v0, f2 + u0 ** 2 + v0 ** 2]], float)
    return float(np.sqrt(float(n2 @ W @ n2) / float(n3 @ W @ n3))), float(np.sqrt(f2))


def cmd_aspect(a):
    img = pl.imread(a.image)
    q = np.float32(a.corners).reshape(4, 2)
    c = pl.order_corners(q)
    edge = lambda i, j: float(np.linalg.norm(c[j] - c[i]))          # noqa: E731
    by_sides = max(edge(0, 1), edge(3, 2)) / max(edge(0, 3), edge(1, 2))
    ratio, f = aspect_from_quad(q, img.shape)
    rep = {"experiment": "aspect", "image": a.image,
           "sides_px": {"top": round(edge(0, 1), 1), "bottom": round(edge(3, 2), 1),
                        "left": round(edge(0, 3), 1), "right": round(edge(1, 2), 1)},
           "aspect_by_sides": round(by_sides, 3),
           "aspect_from_perspective": None if ratio is None else round(ratio, 3),
           "focal_px": None if f is None else round(f),
           "error_of_sides_pct": None if ratio is None else round(100 * (by_sides / ratio - 1), 1)}
    if a.long_mm:                                                   # известна длинная сторона — выводим вторую
        rep["cover_mm"] = [round(a.long_mm * ratio), a.long_mm] if ratio else None
    print(json.dumps(rep, ensure_ascii=False))


def main(argv=None):
    p = argparse.ArgumentParser(prog="experiments", description="опыты для REPORT.md: HSV против RGB, разрыв оттенка, пропорции из перспективы")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("rgb-vs-hsv", help="границы с одного прямоугольника по HSV и по B, G, R на двух кадрах сцены")
    s.add_argument("--day", required=True); s.add_argument("--lamp", required=True)
    s.add_argument("--sample", type=_triple, default=(160, 530, 60, 60), help="x,y,w,h — прямоугольник на объекте")
    s.add_argument("--circle", type=_triple, default=(190, 560, 58), help="x,y,r — эталонный круг")
    s.set_defaults(fn=cmd_rgb_vs_hsv)

    s = sub.add_parser("hue-shift", help="сдвинуть оттенок кадра так, чтобы объект встал на разрыв 0/179")
    s.add_argument("image"); s.add_argument("--shift", type=int, default=27)
    s.add_argument("--sample", type=_triple, default=(160, 530, 60, 60))
    s.add_argument("--circle", type=_triple, default=(190, 560, 58))
    s.add_argument("-o", "--out", required=True)
    s.set_defaults(fn=cmd_hue_shift)

    s = sub.add_parser("lecture-check", help="сверка с числами лекций L3 (демо 1.4) и L4 (стр. 19, демо 2.4)")
    s.add_argument("image", help="кадр лекций: lectures/L05-filtering/demo/iu-building.jpg")
    s.set_defaults(fn=cmd_lecture_check)

    s = sub.add_parser("wb-reference", help="что баланс белого делает с серой опорной поверхностью кадра")
    s.add_argument("image")
    s.add_argument("--max-sat", type=int, default=60, help="порог насыщенности «серого» (S в HSV)")
    s.add_argument("--bright-pct", type=float, default=85.0, help="процентиль яркости: опора — светлые пиксели")
    s.set_defaults(fn=cmd_wb_reference)

    s = sub.add_parser("aspect", help="W/H объекта из перспективы — против наивного «по сторонам в кадре»")
    s.add_argument("image")
    s.add_argument("--corners", required=True, type=lambda v: [float(t) for t in v.split(",")],
                   help="x1,y1,…,x4,y4 в любом порядке")
    s.add_argument("--long-mm", type=float, help="известная длинная сторона в мм — тогда печатается и вторая")
    s.set_defaults(fn=cmd_aspect)

    a = p.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
    sys.exit(main())
