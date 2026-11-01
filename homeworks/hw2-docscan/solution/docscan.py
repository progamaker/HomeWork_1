# -*- coding: utf-8 -*-
"""ДЗ 2 «Сканер документов» — реализация.

Запуск:  python docscan.py <команда> [аргументы]      (список команд — README.md)

Тринадцать функций реализованы на NumPy и OpenCV; всё остальное (разбор
аргументов, чтение и запись, склейка этапов, разметка кликами, прогон по
набору, цикл видео) дано готовым. Склейку ``detect`` можно менять и улучшать;
Имена и сигнатуры функций сохранены для совместимости с тестами.
Самопроверка: ``pytest tests -q``.
"""
import argparse
import csv
import glob
import io
import itertools
import json
import os
import sys
import time

import cv2
import numpy as np

__version__ = "1.0"

A4 = 210.0 / 297.0                                  # W / H листа A4 в книжной ориентации
NOISE_KERNEL = np.float32([[1, -2, 1], [-2, 4, -2], [1, -2, 1]])   # разность двух лапласианов: Σh = 0, Σh² = 36

# --------------------------------------------------------------- ввод-вывод ---


def imread(path):
    """Читает BGR uint8 и падает с внятной ошибкой (cv2.imread молча даёт None; кириллица в пути)."""
    if not os.path.exists(path):
        raise FileNotFoundError("файл не найден: %s" % path)
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("не удалось декодировать изображение: %s" % path)
    return img


def imwrite(path, img):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if not ok:
        raise ValueError("не удалось закодировать %s" % path)
    buf.tofile(path)
    return path


def as_uint8(x):
    """float → uint8 только через обрезку (правило курса: uint8 переполняется молча)."""
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def work_image(img, long_side=640):
    """Кадр → серый рабочего размера (длинная сторона ≤ long_side, INTER_AREA — правило L4) и масштаб.

    Лист ищем на уменьшенном кадре: текст сливается в серое, границы листа остаются, а счёт
    дешевеет вчетверо на каждое уменьшение вдвое — без этого не будет 10 кадров в секунду.
    """
    h, w = img.shape[:2]
    scale = min(1.0, float(long_side) / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return to_gray(img), scale


def to_full(pts, scale):
    """Координаты рабочего кадра → исходного: центр пикселя — в (x + 0.5), отсюда поправка на полпикселя (L4)."""
    return (np.asarray(pts, np.float32) + 0.5) / scale - 0.5


# ------------------------------------------------------------ 1. предобработка ---


def estimate_noise(gray):
    """Оценка σ шума по одному кадру, без эталона (в кодах яркости).

    Кадр сворачивается с NOISE_KERNEL — ядро гасит постоянную, наклон и прямые края вдоль осей,
    шум проходит: у белого шума σ отклик имеет σ_r = σ · sqrt(Σh²) = 6σ (формула L5).
    σ_r оцениваем робастно, через медиану модуля: для нормального закона median|r| = 0.6745 · σ_r —
    так текст и края (выбросы) оценку почти не сдвигают.
    """
    g = gray.astype(np.float32)
    response = cv2.filter2D(g, -1, NOISE_KERNEL)[1:-1, 1:-1]
    return float(np.median(np.abs(response)) / (0.6745 * 6))


def denoise(gray, noise_sigma, target=1.0, lo=1.0, hi=3.0):
    """Сглаживание Гауссом, σ фильтра — по измеренному шуму. Возвращает (кадр, σ фильтра).

    Гаусс с σ_f оставляет от шума σ_n · sqrt(Σh²) ≈ σ_n / (2 · sqrt(π) · σ_f) (L5). Отсюда σ_f, при которой
    остаётся target кодов; меньше lo не берём (Canny на несглаженном кадре ловит зерно), больше hi — тоже (углы листа скругляются).
    """
    sigma_f = float(np.clip(noise_sigma / (2 * np.sqrt(np.pi) * target), lo, hi))
    return cv2.GaussianBlur(gray, (0, 0), sigma_f), sigma_f


def noise_after_filter(sigma_n, sigma_f):
    """Предсказание L5: σ шума после Гаусса с σ_f — по сумме квадратов отсчётов настоящего ядра OpenCV."""
    k = cv2.getGaussianKernel(2 * int(np.ceil(4 * sigma_f)) + 1, sigma_f)      # с запасом шире, чем берёт GaussianBlur
    h = k @ k.T
    return float(sigma_n * np.sqrt((h ** 2).sum()))


def erase_text(gray, k=9):
    """Серое закрытие (L3) ядром k × k стирает тёмные детали уже k px — строки текста, линии таблицы, волокна дерева.

    Границу «светлый лист / тёмный стол» закрытие не сдвигает и не размывает. k = 0 — шаг выключен:
    сравните карту краёв с ним и без него (команда edges, ключ --close).
    """
    return cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8)) if k and k > 1 else gray


# ------------------------------------------------------------------ 2. края ---


def auto_canny(gray, k=0.33):
    """Canny с порогами от медианы яркости кадра: lo = (1 − k) · m, hi = (1 + k) · m. Возвращает (края 0/255, (lo, hi)).

    Правило из туториалов — в отчёте его нужно проверить измерением на своих кадрах, а не принять на веру.
    """
    m = float(np.median(gray))
    lo = int(np.clip((1 - k) * m, 0, 255))
    hi = int(np.clip((1 + k) * m, 0, 255))
    return cv2.Canny(gray, lo, hi, L2gradient=True), (lo, hi)


# ------------------------------------------------------- 3a. путь «контуры» ---


def quad_from_contours(edges, min_area=0.10, eps=0.02, close_k=5, top=5):
    """Карта краёв → четыре угла листа float32 (4, 2) или None.

    Закрытие (чтобы граница листа замкнулась) → findContours → крупнейшие по площади → approxPolyDP
    с допуском eps · периметр → первый выпуклый четырёхугольник площадью ≥ min_area кадра.
    """
    h, w = edges.shape[:2]
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8)) if close_k > 1 else edges
    contours = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:top]:
        quad = cv2.approxPolyDP(contour, eps * cv2.arcLength(contour, True), True)
        if len(quad) == 4 and cv2.isContourConvex(quad) and cv2.contourArea(quad) >= min_area * h * w:
            return order_corners(quad.reshape(4, 2))
    return None


# --------------------------------------------------- 3b. путь «Хаф + RANSAC» ---


def hough_segments(edges, threshold=25, min_len=0.08, max_gap=10):
    """Отрезки HoughLinesP как int32 (N, 4): x1, y1, x2, y2. Форма приводится версионно-нейтрально (в OpenCV 5 она другая)."""
    h, w = edges.shape[:2]
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=int(min_len * min(h, w)), maxLineGap=max_gap)
    return np.zeros((0, 4), np.int32) if segs is None else np.int32(segs).reshape(-1, 4)


def segment_points(segs, step=3.0):
    """Точки вдоль отрезков с шагом step px, float32 (M, 2): длинный отрезок даёт больше точек — больше «голосов»."""
    out = []
    for x1, y1, x2, y2 in np.asarray(segs, np.float32).reshape(-1, 4):
        n = max(2, int(np.hypot(x2 - x1, y2 - y1) / step) + 1)
        t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
        out.append((1 - t) * (x1, y1) + t * (x2, y2))
    return np.concatenate(out).astype(np.float32) if out else np.zeros((0, 2), np.float32)


def line_through(p, q):
    """Прямая через две точки в однородных координатах: (a, b, c), a·x + b·y + c = 0, нормирована так, что a² + b² = 1.

    Тогда |a·x + b·y + c| — расстояние от точки до прямой в пикселях. Совпавшие точки → None.
    """
    line = np.cross(np.r_[np.asarray(p, dtype=float), 1.0], np.r_[np.asarray(q, dtype=float), 1.0])
    norm = np.hypot(line[0], line[1])
    return None if norm < 1e-9 else line / norm


def intersect(l1, l2):
    """Точка пересечения двух прямых (x, y) — векторное произведение (L4); параллельные → None."""
    point = np.cross(l1, l2)
    return None if abs(point[2]) < 1e-9 else point[:2] / point[2]


def point_line_dist(pts, line):
    """Расстояния точек (N, 2) до нормированной прямой (a, b, c), px."""
    return np.abs(pts[:, 0] * line[0] + pts[:, 1] * line[1] + line[2])


def fit_line(pts):
    """Прямая по N ≥ 2 точкам методом наименьших квадратов по расстоянию (total least squares): нормаль — младший сингулярный вектор."""
    c = pts.mean(0)
    n = np.linalg.svd((pts - c).astype(np.float64), full_matrices=False)[2][-1]
    return np.float64([n[0], n[1], -float(n @ c)])


def ransac_line(pts, thr=2.0, iters=200, rng=None):
    """Одна прямая по точкам с выбросами. Возвращает (прямая (a, b, c), булева маска инлаеров) или (None, None).

    iters раз: две случайные точки → line_through → расстояния → число точек ближе thr; лучшая гипотеза
    уточняется fit_line по своим инлаерам, маска пересчитывается. 200 итераций — с запасом для доли
    инлаеров w ≥ 0.15: N = log(1 − 0.99) / log(1 − w²) (L7).
    """
    rng = rng or np.random.default_rng(0)
    n = len(pts)
    if n < 2:
        return None, None
    pts = np.asarray(pts, dtype=np.float64)
    best, best_count = None, 1
    for _ in range(iters):
        i, j = rng.integers(0, n, 2)
        line = line_through(pts[i], pts[j])
        if line is None:
            continue
        inliers = point_line_dist(pts, line) < thr
        count = int(inliers.sum())
        if count > best_count:
            best, best_count = inliers, count
    if best is None:
        return None, None
    line = fit_line(pts[best])
    return line, point_line_dist(pts, line) < thr


def dominant_lines(pts, n_lines=4, thr=2.0, iters=200, min_inliers=25, clear=3.0, rng=None):
    """Последовательный RANSAC: до n_lines прямых, у каждой ≥ min_inliers точек. Возвращает список (a, b, c) — от самой сильной.

    Нашли прямую → убрали её точки и всё, что ближе clear · thr (иначе второй «доминирующей» станет её же двойник) → ищем следующую.
    """
    rng = rng or np.random.default_rng(0)
    remaining = np.asarray(pts, dtype=np.float64)
    lines = []
    for _ in range(n_lines):
        line, inliers = ransac_line(remaining, thr, iters, rng)
        if line is None or inliers.sum() < min_inliers:
            break
        lines.append(line)
        remaining = remaining[point_line_dist(remaining, line) >= clear * thr]
    return lines


def is_convex_quad(q):
    """Четыре точки в циклическом порядке образуют выпуклый четырёхугольник (все повороты — в одну сторону)."""
    d = np.roll(q, -1, axis=0) - q
    z = d[:, 0] * np.roll(d, -1, axis=0)[:, 1] - d[:, 1] * np.roll(d, -1, axis=0)[:, 0]
    return bool(np.all(z > 0) or np.all(z < 0))


def quad_from_lines(lines, shape, margin=0.05, min_area=0.10):
    """Четыре (или больше) прямые → углы листа float32 (4, 2) или None.

    У листа стороны идут парами; четыре прямые делятся на две пары тремя способами. Углы листа — пересечения
    прямых из разных пар. Верное разбиение узнаётся по тому, что все четыре пересечения лежат в кадре
    (с запасом margin): прямые одной пары — противоположные стороны, они сходятся далеко за кадром
    (точка схода, L4). Плюс выпуклость и площадь ≥ min_area кадра. Прямых больше четырёх — перебор четвёрок
    от самых сильных.
    """
    h, w = shape[:2]
    for ids in itertools.combinations(range(len(lines)), 4):
        a, b, c, d = [lines[i] for i in ids]
        for pair1, pair2 in (((a, b), (c, d)), ((a, c), (b, d)), ((a, d), (b, c))):
            points = [intersect(l1, l2) for l1 in pair1 for l2 in pair2]
            if any(point is None for point in points):
                continue
            q = np.asarray(points, dtype=np.float32)
            if not np.isfinite(q).all():
                continue
            if np.any(q < (-margin * w, -margin * h)) or np.any(q > ((1 + margin) * w, (1 + margin) * h)):
                continue
            q = order_corners(q)
            if is_convex_quad(q) and cv2.contourArea(q) >= min_area * h * w:
                return q
    return None


def quad_from_hough(edges, n_lines=4, thr=2.0, iters=200, seed=0):
    """Карта краёв → отрезки Хафа → точки → четыре доминирующие прямые (RANSAC) → их пересечения."""
    segs = hough_segments(edges)
    if len(segs) < 4:
        return None
    lines = dominant_lines(segment_points(segs), n_lines, thr, iters, rng=np.random.default_rng(seed))
    return quad_from_lines(lines, edges.shape) if len(lines) >= 4 else None


# ------------------------------------------------------------- 4. геометрия ---


def order_corners(pts):
    """Четыре точки в любом порядке → float32 (4, 2) по часовой стрелке, первым — угол с минимальной x + y.

    Для листа, снятого «примерно ровно», это tl, tr, br, bl. Порядок — по углу вокруг центроида: правило
    «суммы и разности» из занятия 4 при повороте листа на 45° даёт «бабочку», а сканеру лист кладут как угодно.
    """
    q = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    offsets = q - q.mean(axis=0)
    q = q[np.argsort(np.arctan2(offsets[:, 1], offsets[:, 0]))]
    return np.roll(q, -int(np.argmin(q.sum(axis=1))), axis=0)


def rectify(img, corners, size=None, aspect=None):
    """Четыре угла → (фронтальный вид, H). Размер: size = (W, H), иначе из aspect = W / H, иначе из длин сторон в кадре.

    С aspect ширина берётся такой, чтобы ни одна сторона листа не уменьшалась: W = max(верх, низ, aspect · лево, aspect · право).
    """
    q = order_corners(corners)
    if size is None:
        top, right, bottom, left = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)
        width, height = max(top, bottom), max(left, right)
        if aspect is not None:
            if not np.isfinite(aspect) or aspect <= 0:
                raise ValueError("aspect должен быть положительным")
            width = max(width, aspect * height)
            height = width / aspect
        size = (max(2, int(round(float(width)))), max(2, int(round(float(height)))))
    width, height = map(int, size)
    dst = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    matrix = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(img, matrix, (width, height), borderMode=cv2.BORDER_REPLICATE), matrix


def quad_iou(q1, q2, shape):
    """IoU двух четырёхугольников в кадре shape — через две маски. None вместо четырёхугольника → 0.0."""
    if q1 is None or q2 is None:
        return 0.0
    masks = [np.zeros(shape[:2], np.uint8) for _ in range(2)]
    for mask, quad in zip(masks, (q1, q2)):
        cv2.fillConvexPoly(mask, np.int32(np.round(order_corners(quad))), 1)
    union = np.count_nonzero(masks[0] | masks[1])
    return float(np.count_nonzero(masks[0] & masks[1]) / union) if union else 0.0


# ------------------------------------------------------ 5. тени и бинаризация ---


def remove_shadows(gray, k=51):
    """Выравнивание освещения: страница ÷ оценка фона · 255 (uint8). Фон — то, что осталось бы без чернил.

    Медиана по окну k × k штрихов не замечает (в окне их меньше половины), а тень — медленное изменение —
    в ней остаётся и при делении сокращается: свет входит в яркость множителем (занятие 3, «страница ÷ размытый фон»).
    """
    background = cv2.medianBlur(gray, k).astype(np.float32)
    return as_uint8(255 * gray.astype(np.float32) / np.maximum(background, 1))


def binarize(gray, method="flat", block=31, C=15, k=51):
    """Чёрно-белый скан (0 — чернила, 255 — бумага): ``otsu`` — один порог на кадр (L3); ``adaptive`` — порог по окну
    block со сдвигом C (L3); ``flat`` — remove_shadows, затем Otsu."""
    if method == "flat":
        gray = remove_shadows(gray, k)
    if method in ("otsu", "flat"):
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if method == "adaptive":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, C)
    raise ValueError("метод бинаризации: otsu, adaptive или flat, а не %r" % (method,))


def ink_match(binary, gt):
    """Доля пикселей (%), где «чернила / бумага» совпали с эталоном; эталон приводится к размеру скана."""
    gt = to_gray(gt)
    if gt.shape != binary.shape:
        gt = cv2.resize(gt, (binary.shape[1], binary.shape[0]), interpolation=cv2.INTER_AREA)
    return float(100.0 * ((binary < 128) == (gt < 128)).mean())


# ---------------------------------------------------------------- пайплайн ---


def looks_like_paper(gray, quad, ring=15, min_gain=1.05):
    """Проверка «это лист, а не книга и не монитор»: медиана яркости внутри четырёхугольника выше, чем в кольце вокруг него."""
    inside = np.zeros(gray.shape[:2], np.uint8)
    cv2.fillConvexPoly(inside, np.int32(np.round(quad)), 1)
    around = cv2.dilate(inside, np.ones((2 * ring + 1, 2 * ring + 1), np.uint8)) - inside
    if not inside.any() or not around.any():
        return True
    return float(np.median(gray[inside > 0])) >= min_gain * float(np.median(gray[around > 0]))


def edge_map(gray, canny="auto", k=0.33, close_k=9):
    """Серый рабочий кадр → (края, числа этапа): оценка шума → Гаусс по шуму → закрытие текста → Canny."""
    sigma_n = estimate_noise(gray)
    smooth, sigma_f = denoise(gray, sigma_n)
    smooth = erase_text(smooth, close_k)
    if canny == "auto":
        edges, (lo, hi) = auto_canny(smooth, k)
    else:
        lo, hi = int(canny[0]), int(canny[1])
        edges = cv2.Canny(smooth, lo, hi, L2gradient=True)
    return edges, {"noise_sigma": round(sigma_n, 2), "filter_sigma": round(sigma_f, 2), "median": float(np.median(smooth)),
                   "canny": [lo, hi], "edge_pct": round(float(100.0 * (edges > 0).mean()), 2)}


def detect(img, method="auto", long_side=640, canny="auto", k=0.33, close_k=9, paper_check=True, seed=0):
    """Кадр → (углы листа в координатах кадра, по часовой от верхнего левого, или None; числа этапов).

    method: ``contour`` — путь 3a, ``hough`` — путь 3b, ``auto`` — сначала контуры, не нашли — Хаф.
    canny: ``auto`` (пороги от медианы, параметр k) или пара (lo, hi). close_k — ядро закрытия текста (0 — без него).
    Эту склейку можно менять и улучшать — сигнатуры функций с TODO менять нельзя.
    """
    t0 = time.perf_counter()
    gray, scale = work_image(img, long_side)
    edges, info = edge_map(gray, canny, k, close_k)
    quad, path = None, None
    for m in (("contour", "hough") if method == "auto" else (method,)):
        q = quad_from_contours(edges) if m == "contour" else quad_from_hough(edges, seed=seed)
        if q is not None and (not paper_check or looks_like_paper(gray, q)):
            quad, path = order_corners(to_full(q, scale)), m
            break
    info.update(path=path, ms=round(1000.0 * (time.perf_counter() - t0), 1))
    return quad, info


def draw_quad(img, quad, colour=(0, 220, 0), label=None):
    out = img.copy()
    t = max(2, int(round(max(img.shape[:2]) / 400)))
    if quad is not None:
        q = np.int32(np.round(quad))
        cv2.polylines(out, [q], True, colour, t, cv2.LINE_AA)
        for i, p in enumerate(q):
            cv2.circle(out, (int(p[0]), int(p[1])), 3 * t, (0, 0, 255) if i == 0 else colour, -1, cv2.LINE_AA)
    if label:
        cv2.putText(out, label, (12, 14 + 12 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * t, (0, 0, 0), 2 * t, cv2.LINE_AA)
        cv2.putText(out, label, (12, 14 + 12 * t), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * t, (255, 255, 255), max(1, t // 2), cv2.LINE_AA)
    return out


def load_annotations(path):
    """Разметка: JSON {имя файла: [[x, y] × 4] или null (листа в кадре нет)}."""
    if not path or not os.path.exists(path):
        return {}
    with io.open(path, encoding="utf-8") as f:
        return json.load(f)


def save_annotations(path, ann):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(ann, f, ensure_ascii=False, indent=1)


def pick_corners(img, title="4 klika po uglam lista | Enter - gotovo | r - zanovo | n - lista net"):
    """Окно OpenCV: четыре клика по углам → float32 (4, 2); клавиша n — «листа нет» (None). Только локально."""
    pts, scale, state = [], min(1.0, 1200.0 / img.shape[1]), {"none": False}

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x / scale, y / scale))

    cv2.namedWindow(title); cv2.setMouseCallback(title, on_click)
    while True:
        view = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for p in pts:
            cv2.circle(view, (int(p[0] * scale), int(p[1] * scale)), 6, (0, 200, 255), -1)
        cv2.imshow(title, view)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("r"):
            pts.clear()
        if key == ord("n"):
            state["none"] = True
            break
        if key in (13, 10, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    if state["none"]:
        return None
    if len(pts) != 4:
        raise ValueError("нужно ровно 4 клика, получено %d" % len(pts))
    return order_corners(pts)


def evaluate(paths, ann, methods=("contour", "hough", "auto"), iou_ok=0.9, seeds=1, overlays=None, **kw):
    """Прогон по набору. Возвращает (строки по кадрам, сводку по методам).

    Кадр с листом: успех — IoU ≥ iou_ok. Кадр без листа (в разметке null): успех — «не найдено».
    seeds > 1 — путь hough прогоняется с зёрнами 0…seeds − 1 (RANSAC случаен): в строке — число успехов.
    overlays — каталог, куда сложить кадры с найденным (зелёный) и размеченным (синий) четырёхугольником.
    """
    rows = []
    kw = dict(kw); seed0 = kw.pop("seed", 0)
    for p in paths:
        name = os.path.basename(p)
        if name not in ann:
            continue
        img, gt = imread(p), ann[name]
        gt_q = np.float32(gt) if gt is not None else None
        ok = lambda q: (q is None) if gt is None else quad_iou(q, gt_q, img.shape) >= iou_ok      # noqa: E731
        row = {"file": name, "sheet": gt is not None}
        for m in methods:
            quad, info = detect(img, m, seed=seed0, **kw)
            row[m + "_found"] = quad is not None
            row[m + "_iou"] = round(quad_iou(quad, gt_q, img.shape), 3) if gt is not None else None
            row[m + "_ms"] = info["ms"]
            if overlays:
                view = draw_quad(img, quad, label="%s: %s" % (m, info["path"] or "net lista"))
                if gt is not None:
                    cv2.polylines(view, [np.int32(gt_q)], True, (255, 160, 0), 1, cv2.LINE_AA)
                imwrite(os.path.join(overlays, "%s_%s.jpg" % (os.path.splitext(name)[0], m)), view)
        row["auto_path"] = info["path"]
        if seeds > 1 and "hough" in methods:
            row["hough_ok_of_%d" % seeds] = int(sum(ok(detect(img, "hough", seed=sd, **kw)[0]) for sd in range(seeds)))
        rows.append(row)
    summary = {}
    pos, neg = [r for r in rows if r["sheet"]], [r for r in rows if not r["sheet"]]
    for m in methods:
        summary[m] = {"sheets": len(pos), "found_iou_ok": sum(r[m + "_iou"] >= iou_ok for r in pos),
                      "mean_iou": round(float(np.mean([r[m + "_iou"] for r in pos])), 3) if pos else None,
                      "no_sheet": len(neg), "rejected": sum(not r[m + "_found"] for r in neg),
                      "mean_ms": round(float(np.mean([r[m + "_ms"] for r in rows])), 1) if rows else None}
    return rows, summary


def frames_from(source):
    """Кадры из камеры (номер), видеофайла (путь) или готового списка кадров."""
    if isinstance(source, (list, tuple)):
        for f in source:
            yield f
        return
    cap = cv2.VideoCapture(int(source)) if str(source).isdigit() else cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError("не открывается источник видео: %s (камера занята? Colab? — нужен локальный запуск)" % (source,))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def run_video(source, method="auto", seconds=10.0, show=True, save=None, max_frames=None, **kw):
    """Видеорежим: лист ищется в каждом кадре. Возвращает числа: кадров, средний FPS всего цикла, доля кадров с листом, мс на detect."""
    n = found = 0
    ms, writer = [], None
    t_start = time.perf_counter()
    for frame in frames_from(source):
        quad, info = detect(frame, method, **kw)
        n += 1; found += quad is not None; ms.append(info["ms"])
        fps = n / max(time.perf_counter() - t_start, 1e-6)
        if show or save:
            view = draw_quad(frame, quad, label="%s  %.1f FPS  %.0f ms" % (info["path"] or "net lista", fps, info["ms"]))
            if save:
                if writer is None:
                    os.makedirs(os.path.dirname(os.path.abspath(save)), exist_ok=True)
                    writer = cv2.VideoWriter(save, cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (frame.shape[1], frame.shape[0]))
                writer.write(view)
            if show:
                cv2.imshow("docscan: q - vyhod", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        if (seconds and time.perf_counter() - t_start >= seconds) or (max_frames and n >= max_frames):
            break
    elapsed = time.perf_counter() - t_start
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    return {"frames": n, "seconds": round(elapsed, 2), "fps": round(n / elapsed, 1) if elapsed > 0 and n else 0.0,
            "found_pct": round(100.0 * found / n, 1) if n else 0.0, "detect_ms_median": round(float(np.median(ms)), 1) if ms else None}


# --------------------------------------------------------------------- CLI ---


def _log(**kw):
    print(json.dumps(kw, ensure_ascii=False))


def _canny_arg(s):
    return "auto" if s == "auto" else tuple(int(v) for v in s.split(","))


def _corners_list(q):
    return None if q is None else [[round(float(x), 1), round(float(y), 1)] for x, y in q]


def _detect_kw(a):
    return dict(long_side=a.long_side, canny=a.canny, k=a.k, close_k=a.close, paper_check=not a.no_paper_check, seed=a.seed)


def _gt_for(a, path):
    ann = load_annotations(a.ann) if getattr(a, "ann", None) else {}
    return ann.get(os.path.basename(path), "нет")


def cmd_noise(a):
    img = imread(a.image); gray, scale = work_image(img, a.long_side)
    rep = {"command": "noise", "work_size": [gray.shape[1], gray.shape[0]], "noise_sigma": round(estimate_noise(gray), 2)}
    if a.add_sigma:
        clean = gray.astype(np.float32)
        noisy = clean + np.random.default_rng(a.seed).normal(0, a.add_sigma, gray.shape).astype(np.float32)
        est = estimate_noise(as_uint8(noisy))
        sigma_f = denoise(as_uint8(noisy), est)[1]
        left = cv2.GaussianBlur(noisy, (0, 0), sigma_f) - cv2.GaussianBlur(clean, (0, 0), sigma_f)      # фильтр линеен: разность — отфильтрованный шум
        rep.update(added_sigma=a.add_sigma, noise_sigma_noisy=round(est, 2), filter_sigma=round(sigma_f, 2),
                   noise_after_measured=round(float(left.std()), 2), noise_after_predicted=round(noise_after_filter(a.add_sigma, sigma_f), 2))
    _log(**rep)


def cmd_edges(a):
    img = imread(a.image); gray, scale = work_image(img, a.long_side)
    edges, info = edge_map(gray, a.canny, a.k, a.close)
    imwrite(a.out, edges)
    _log(command="edges", close=a.close, **info, segments=int(len(hough_segments(edges))), out=a.out)


def cmd_detect(a):
    img = imread(a.image)
    quad, info = detect(img, a.method, **_detect_kw(a))
    rep = {"command": "detect", "method": a.method, "corners": _corners_list(quad), **info}
    gt = _gt_for(a, a.image)
    if gt != "нет":
        rep["iou"] = round(quad_iou(quad, np.float32(gt), img.shape), 3) if gt is not None else None
        rep["sheet_in_gt"] = gt is not None
    if a.out:
        view = draw_quad(img, quad, label="%s  %s" % (a.method, info["path"] or "net lista"))
        if gt not in ("нет", None):
            cv2.polylines(view, [np.int32(gt)], True, (255, 160, 0), 1, cv2.LINE_AA)
        imwrite(a.out, view); rep["out"] = a.out
    _log(**rep)


def cmd_scan(a):
    img = imread(a.image)
    if a.corners:
        quad, info = order_corners(np.float32(a.corners).reshape(4, 2)), {"path": "corners"}
    else:
        quad, info = detect(img, a.method, **_detect_kw(a))
    if quad is None:
        _log(command="scan", found=False, **info)
        raise SystemExit("лист не найден — скан не построен (попробуйте другие пороги: --canny 50,150; другой путь: --method hough; или углы руками: --corners)")
    front, _ = rectify(img, quad, aspect=a.aspect if a.turn % 2 == 0 else 1.0 / a.aspect)     # лист лежит боком — выпрямляем «альбомно», потом поворачиваем
    front = np.ascontiguousarray(np.rot90(front, -a.turn)) if a.turn else front
    bw = binarize(to_gray(front), a.binarize, a.block, a.C)
    imwrite(a.out, front)
    rep = {"command": "scan", "found": True, "corners": _corners_list(quad), "size_out": [front.shape[1], front.shape[0]], "binarize": a.binarize, "out": a.out, **info}
    if a.bw:
        imwrite(a.bw, bw); rep["bw"] = a.bw
    if a.gt:
        rep["ink_match_pct"] = round(ink_match(bw, imread(a.gt)), 1)
    _log(**rep)


def cmd_binarize(a):
    gray = to_gray(imread(a.image))
    rep = {"command": "binarize", "block": a.block, "C": a.C}
    methods = ("otsu", "adaptive", "flat") if a.method == "all" else (a.method,)
    gt = imread(a.gt) if a.gt else None
    for m in methods:
        bw = binarize(gray, m, a.block, a.C)
        rep[m] = {"ink_pct": round(float(100.0 * (bw < 128).mean()), 2)}
        if gt is not None:
            rep[m]["match_pct"] = round(ink_match(bw, gt), 1)
        if a.out:
            root, ext = os.path.splitext(a.out)
            imwrite(a.out if len(methods) == 1 else "%s_%s%s" % (root, m, ext), bw)
    _log(**rep)


def _expand(patterns):
    out = []
    for p in patterns:
        if os.path.isdir(p):
            for e in ("*.jpg", "*.jpeg", "*.png"):
                out += glob.glob(os.path.join(p, e))
        else:
            out += glob.glob(p) or [p]
    return sorted(set(out))


def cmd_annotate(a):
    ann = load_annotations(a.ann)
    for p in _expand(a.images):
        name = os.path.basename(p)
        if name in ann and not a.redo:
            continue
        quad = pick_corners(imread(p))
        ann[name] = _corners_list(quad)
        save_annotations(a.ann, ann)
        _log(command="annotate", file=name, corners=ann[name])


def cmd_eval(a):
    ann = load_annotations(a.ann)
    if not ann:
        raise SystemExit("eval: разметка %s пуста или не найдена — сначала annotate" % a.ann)
    rows, summary = evaluate(_expand(a.images), ann, iou_ok=a.iou_ok, seeds=a.seeds, overlays=a.overlays, **_detect_kw(a))
    if not rows:
        raise SystemExit("eval: ни один файл из списка не найден в разметке %s" % a.ann)
    if a.csv:
        os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
        with io.open(a.csv, "w", encoding="utf-8", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    for r in rows:
        _log(**r)
    _log(command="eval", n=len(rows), iou_ok=a.iou_ok, canny=a.canny if a.canny == "auto" else list(a.canny), summary=summary, csv=a.csv)


def cmd_video(a):
    stats = run_video(a.source, a.method, a.seconds, not a.no_window, a.save, **_detect_kw(a))
    _log(command="video", source=a.source, method=a.method, **stats)


def build_parser():
    p = argparse.ArgumentParser(prog="docscan", description="ДЗ 2 «Сканер документов»: шум, края, лист двумя путями, выпрямление, бинаризация, видео")
    sub = p.add_subparsers(dest="command", required=True)

    def common(s):
        s.add_argument("--long-side", type=int, default=640, help="длинная сторона рабочего кадра, px")
        s.add_argument("--canny", type=_canny_arg, default="auto", help="auto (пороги от медианы) или LO,HI")
        s.add_argument("--k", type=float, default=0.33, help="ширина вилки порогов вокруг медианы")
        s.add_argument("--close", type=int, default=9, help="ядро закрытия, стирающего текст (0 — выключить)")
        s.add_argument("--no-paper-check", action="store_true", help="не проверять, что внутри четырёхугольника светлее, чем вокруг")
        s.add_argument("--seed", type=int, default=0, help="зерно RANSAC")

    s = sub.add_parser("noise", help="оценка шума кадра и проверка фильтра"); s.add_argument("image"); s.add_argument("--long-side", type=int, default=640)
    s.add_argument("--add-sigma", type=float, help="добавить гауссов шум с известной σ и проверить оценку и формулу σ·sqrt(Σh²)"); s.add_argument("--seed", type=int, default=0); s.set_defaults(fn=cmd_noise)

    s = sub.add_parser("edges", help="карта краёв Canny"); s.add_argument("image"); common(s); s.add_argument("-o", "--out", required=True); s.set_defaults(fn=cmd_edges)

    s = sub.add_parser("detect", help="углы листа"); s.add_argument("image"); common(s)
    s.add_argument("--method", choices=("contour", "hough", "auto"), default="auto"); s.add_argument("--ann", help="разметка corners.json — будет посчитан IoU")
    s.add_argument("-o", "--out", help="кадр с нарисованным четырёхугольником"); s.set_defaults(fn=cmd_detect)

    s = sub.add_parser("scan", help="полный пайплайн: лист → выпрямленный скан"); s.add_argument("image"); common(s)
    s.add_argument("--method", choices=("contour", "hough", "auto"), default="auto")
    s.add_argument("--corners", type=lambda v: [float(t) for t in v.split(",")], help="углы руками: x1,y1,…,x4,y4 — вместо поиска")
    s.add_argument("--aspect", type=float, default=A4, help="W / H листа: A4 книжный — 0.707, альбомный — 1.414")
    s.add_argument("--turn", type=int, default=0, choices=(0, 1, 2, 3), help="повернуть скан на четверть оборота по часовой (0…3)")
    s.add_argument("--binarize", choices=("otsu", "adaptive", "flat"), default="flat")
    s.add_argument("--block", type=int, default=31); s.add_argument("--C", type=int, default=15)
    s.add_argument("-o", "--out", required=True, help="цветной выпрямленный вид"); s.add_argument("--bw", help="чёрно-белый скан")
    s.add_argument("--gt", help="эталон страницы «в лоб» — будет посчитано совпадение чернил, %%"); s.set_defaults(fn=cmd_scan)

    s = sub.add_parser("binarize", help="бинаризация уже выпрямленной страницы"); s.add_argument("image")
    s.add_argument("--method", choices=("otsu", "adaptive", "flat", "all"), default="all")
    s.add_argument("--block", type=int, default=31); s.add_argument("--C", type=int, default=15)
    s.add_argument("--gt", help="эталонная бинарная страница"); s.add_argument("-o", "--out"); s.set_defaults(fn=cmd_binarize)

    s = sub.add_parser("annotate", help="разметка углов кликами"); s.add_argument("images", nargs="+"); s.add_argument("--ann", default="data/corners.json")
    s.add_argument("--redo", action="store_true", help="переразметить уже размеченные"); s.set_defaults(fn=cmd_annotate)

    s = sub.add_parser("eval", help="IoU и время обоих путей по набору"); s.add_argument("images", nargs="+"); common(s)
    s.add_argument("--ann", default="data/corners.json"); s.add_argument("--iou-ok", type=float, default=0.9); s.add_argument("--csv")
    s.add_argument("--seeds", type=int, default=1, help="сколько зёрен RANSAC проверить на пути hough"); s.add_argument("--overlays", help="каталог для кадров с разметкой")
    s.set_defaults(fn=cmd_eval)

    s = sub.add_parser("video", help="поиск листа в видео"); common(s); s.add_argument("--source", default="0", help="номер камеры или путь к видеофайлу")
    s.add_argument("--method", choices=("contour", "hough", "auto"), default="auto"); s.add_argument("--seconds", type=float, default=10.0)
    s.add_argument("--no-window", action="store_true", help="без окна — только счёт (для видеофайла и CI)"); s.add_argument("--save", help="записать видео с разметкой, .mp4"); s.set_defaults(fn=cmd_video)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
    sys.exit(main())
