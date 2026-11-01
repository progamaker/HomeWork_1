# -*- coding: utf-8 -*-
"""Самопроверка ДЗ 2 «Сканер документов»: ``pytest tests -q`` из корня вашего репозитория.

Данные синтетические — ни ваших фото, ни камеры тестам не нужны. Пока функция
не написана, её тест падает с ``NotImplementedError``: это и есть список дел.
Те же тесты запускает CI и проверяющий; проходят все — половина работы сделана,
вторая половина — ваши пятнадцать кадров, числа и отчёт.
"""
import itertools
import json
import os
import subprocess
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import docscan as ds  # noqa: E402

W, H = 640, 480
QUAD = np.float32([[190, 60], [450, 80], [500, 420], [130, 400]])          # tl, tr, br, bl


# ------------------------------------------------------------- сцены ---
def page_img(w=300, h=424):
    """Страница: девять строк текста и линейка — то, что сканер должен пережить, а не принять за границу листа."""
    page = np.full((h, w, 3), 245, np.uint8)
    for i in range(9):
        cv2.putText(page, "Computer vision %d" % i, (20, 50 + 36 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.line(page, (20, 380), (w - 20, 380), (30, 30, 30), 2)
    return page


def desk_scene(quad=QUAD, cover=None, sheet=True, seed=0, noise=3.0):
    """Кадр 640 × 480: «деревянный» стол, на нём лист под углом; cover — центр предмета, закрывающего угол листа."""
    rng = np.random.default_rng(seed)
    tex = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sigmaX=25, sigmaY=1.2)
    tex /= tex.std()
    img = np.stack([70 + 10 * tex, 95 + 12 * tex, 125 + 14 * tex], -1)
    if sheet:
        page = page_img().astype(np.float32)
        src = np.float32([[0, 0], [299, 0], [299, 423], [0, 423]])
        Hm = cv2.getPerspectiveTransform(src, np.float32(quad))
        m = cv2.warpPerspective(np.ones((424, 300), np.float32), Hm, (W, H))[..., None]
        img = img * (1 - m) + cv2.warpPerspective(page, Hm, (W, H)) * m
    if cover is not None:
        img = np.ascontiguousarray(img)
        cv2.circle(img, cover, 48, (60, 50, 170), -1, cv2.LINE_AA)
    xx = np.arange(W, dtype=np.float32)[None, :, None]
    img = img * (1.0 - 0.3 * xx / W) + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def corners_close(a, b, tol):
    """Два набора из четырёх углов совпадают с точностью tol px — порядок не важен."""
    a, b = np.float32(a).reshape(4, 2), np.float32(b).reshape(4, 2)
    d = np.linalg.norm(a[:, None] - b[None], axis=2)
    return bool(d.min(1).max() <= tol and d.min(0).max() <= tol)


def quad_points(quad, per_side=80, noise=0.4, seed=0):
    """Точки вдоль четырёх сторон четырёхугольника с небольшим разбросом — как точки отрезков Хафа."""
    rng = np.random.default_rng(seed)
    pts = []
    for p, q in zip(quad, np.roll(quad, -1, axis=0)):
        t = rng.uniform(0.03, 0.97, (per_side, 1))
        pts.append(p + t * (q - p) + rng.normal(0, noise, (per_side, 2)))
    return np.float32(np.concatenate(pts))


def dist_to_line(p, line):
    return abs(line[0] * p[0] + line[1] * p[1] + line[2])


# ---------------------------------------------------- 1. предобработка ---
def test_estimate_noise_matches_known_sigma():
    yy, xx = np.mgrid[:240, :320].astype(np.float32)
    clean = 90 + 60 * xx / 320 + 20 * np.sin(yy / 40.0)                   # гладкий кадр без обрезки в 0 и 255
    assert ds.estimate_noise(clean) < 0.3, "на гладком кадре без шума оценка должна быть около нуля"
    for sigma in (3.0, 8.0, 20.0):
        noisy = clean + np.random.default_rng(1).normal(0, sigma, clean.shape)
        est = ds.estimate_noise(noisy.astype(np.float32))
        assert abs(est - sigma) / sigma < 0.08, "σ = %g, оценка %.2f — расхождение больше 8 %%" % (sigma, est)
    assert isinstance(ds.estimate_noise(np.uint8(clean)), float)


def test_estimate_noise_is_robust_to_text():
    page = cv2.cvtColor(page_img(), cv2.COLOR_BGR2GRAY).astype(np.float32) * 0.8       # штрихи текста — выбросы для оценки
    noisy = page + np.random.default_rng(2).normal(0, 6.0, page.shape)
    est = ds.estimate_noise(noisy.astype(np.float32))
    assert abs(est - 6.0) / 6.0 < 0.22, "текст сдвигает оценку слишком сильно (сейчас %.2f при σ = 6): через медиану модуля выходит +15 %%, через среднее +40 %%" % est


def test_denoise_picks_sigma_from_noise():
    g = cv2.cvtColor(desk_scene(), cv2.COLOR_BGR2GRAY)
    out, s = ds.denoise(g, 5.0)
    assert out.shape == g.shape and out.dtype == np.uint8
    assert abs(s - 5.0 / (2 * np.sqrt(np.pi))) < 0.02, "σ фильтра = σ шума / (2·sqrt(π)·target) при target = 1"
    assert np.abs(out.astype(int) - cv2.GaussianBlur(g, (0, 0), s)).max() <= 1, "фильтр — cv2.GaussianBlur с найденной σ"
    assert ds.denoise(g, 0.5)[1] == pytest.approx(1.0), "σ фильтра не меньше lo = 1"
    assert ds.denoise(g, 60.0)[1] == pytest.approx(3.0), "σ фильтра не больше hi = 3"
    assert ds.denoise(g, 8.0, target=2.0)[1] == pytest.approx(8.0 / (4 * np.sqrt(np.pi)), abs=0.02)


# ------------------------------------------------------------ 2. края ---
def test_auto_canny_thresholds_follow_median():
    g = np.full((200, 300), 100, np.uint8); g[50:150, 80:220] = 220
    edges, (lo, hi) = ds.auto_canny(g, 0.33)
    assert edges.shape == g.shape and edges.dtype == np.uint8 and set(np.unique(edges).tolist()) <= {0, 255}
    assert abs(lo - 67) <= 1 and abs(hi - 133) <= 1, "медиана 100, k = 0.33 → пороги 67 и 133 (сейчас %s, %s)" % (lo, hi)
    assert (edges > 0).sum() > 400, "граница прямоугольника должна найтись"
    assert edges[5:40, 5:70].sum() == 0, "на ровном фоне краёв нет"
    bright = np.full((100, 100), 230, np.uint8)
    lo, hi = ds.auto_canny(bright, 0.33)[1]
    assert hi == 255 and abs(lo - 154) <= 1, "верхний порог обрезается в 255"
    assert ds.auto_canny(g, 0.0)[1] == (100, 100)


# -------------------------------------------------- 3a. путь «контуры» ---
def test_quad_from_contours():
    edges = np.zeros((H, W), np.uint8)
    cv2.polylines(edges, [np.int32(QUAD)], True, 255, 1)
    cv2.circle(edges, (560, 90), 30, 255, 1); cv2.line(edges, (20, 450), (300, 470), 255, 1)       # посторонние края
    q = ds.quad_from_contours(edges)
    assert q is not None and np.asarray(q).shape == (4, 2)
    assert corners_close(q, QUAD, 4), "углы листа — с точностью 4 px"
    broken = edges.copy(); broken[200:280, :200] = 0                                               # граница разорвана на 80 px
    assert ds.quad_from_contours(broken) is None, "разорванная граница — не замкнутый контур: None"
    small = np.zeros((H, W), np.uint8); cv2.rectangle(small, (100, 100), (200, 180), 255, 1)
    assert ds.quad_from_contours(small) is None, "четырёхугольник меньше 10 %% кадра — не лист"
    circle = np.zeros((H, W), np.uint8); cv2.circle(circle, (320, 240), 180, 255, 1)
    assert ds.quad_from_contours(circle) is None, "круг — не четырёхугольник"


# ----------------------------------------------- 3b. путь «Хаф + RANSAC» ---
def test_line_through_and_intersect():
    l = ds.line_through((0, 0), (10, 0))
    assert np.hypot(l[0], l[1]) == pytest.approx(1.0), "нормировка: a² + b² = 1"
    assert dist_to_line((5, 3), l) == pytest.approx(3.0), "|a·x + b·y + c| — расстояние в пикселях"
    assert dist_to_line((7, 0), l) == pytest.approx(0.0, abs=1e-9)
    d = ds.line_through((0, 0), (3, 4))
    assert dist_to_line((4, -3), d) == pytest.approx(5.0)
    assert ds.line_through((2, 2), (2, 2)) is None, "совпавшие точки прямую не задают"
    x = ds.intersect(ds.line_through((2, 0), (2, 9)), ds.line_through((0, 5), (9, 5)))
    assert np.allclose(x, (2, 5))
    assert ds.intersect(ds.line_through((0, 0), (1, 1)), ds.line_through((0, 3), (1, 4))) is None, "параллельные прямые: None"
    far = ds.intersect(ds.line_through((0, 0), (100, 1)), ds.line_through((0, 10), (100, 10)))
    assert np.allclose(far, (1000, 10)), "почти параллельные пересекаются далеко — это точка схода, а не ошибка"


def test_ransac_line_ignores_outliers():
    rng = np.random.default_rng(3)
    x = rng.uniform(0, 600, 120)
    good = np.stack([x, 0.5 * x + 20], 1) + rng.normal(0, 0.5, (120, 2))
    bad = rng.uniform(0, 600, (80, 2))
    pts = np.float32(np.concatenate([good, bad]))
    line, inl = ds.ransac_line(pts, thr=2.0, iters=200, rng=np.random.default_rng(0))
    assert line is not None and inl.dtype == bool and inl.shape == (200,)
    assert np.hypot(line[0], line[1]) == pytest.approx(1.0, abs=1e-6)
    assert dist_to_line((0, 20), line) < 1.0 and dist_to_line((600, 320), line) < 1.0, "прямая y = 0.5x + 20 найдена, несмотря на 40 %% выбросов"
    assert inl[:120].sum() >= 110 and inl[120:].sum() <= 8
    lsq = np.polyfit(pts[:, 0], pts[:, 1], 1)
    assert abs(lsq[0] - 0.5) > 0.05, "контроль: обычный МНК на тех же точках сорван выбросами"
    assert ds.ransac_line(pts[:1])[0] is None, "одной точки мало: (None, None)"


def test_dominant_lines_finds_four_sides():
    rng = np.random.default_rng(4)
    pts = np.concatenate([quad_points(QUAD), np.float32(rng.uniform((0, 0), (W, H), (120, 2)))])
    pts = np.concatenate([pts, quad_points(np.float32([[40, 30], [60, 30], [60, 45], [40, 45]]), per_side=4)])   # мелочь: 16 точек
    lines = ds.dominant_lines(pts, n_lines=4, rng=np.random.default_rng(0))
    assert len(lines) == 4
    for p, q in zip(QUAD, np.roll(QUAD, -1, axis=0)):
        best = min(max(dist_to_line(p, l), dist_to_line(q, l)) for l in lines)
        assert best < 2.0, "сторона %s–%s не найдена среди четырёх прямых" % (p.tolist(), q.tolist())
    two = np.concatenate([quad_points(QUAD)[:160], np.float32(rng.uniform((0, 0), (W, H), (60, 2)))])
    few = ds.dominant_lines(two, n_lines=4, rng=np.random.default_rng(0))
    assert len(few) == 2, "точки есть только у двух сторон — прямых две: случайные точки не дотягивают до min_inliers (сейчас прямых %d)" % len(few)
    assert ds.dominant_lines(np.zeros((0, 2), np.float32)) == []


def test_quad_from_lines():
    sides = [ds.line_through(p, q) for p, q in zip(QUAD, np.roll(QUAD, -1, axis=0))]
    for perm in itertools.permutations(range(4)):
        q = ds.quad_from_lines([sides[i] for i in perm], (H, W))
        assert q is not None and corners_close(q, QUAD, 0.5), "порядок прямых %s не должен влиять на результат" % (perm,)
    assert ds.quad_from_lines(sides[:3], (H, W)) is None, "трёх прямых мало"
    big = QUAD * 3 - [400, 300]
    out = [ds.line_through(p, q) for p, q in zip(big, np.roll(big, -1, axis=0))]
    assert ds.quad_from_lines(out, (H, W)) is None, "углы далеко за кадром — это не лист"
    tiny = np.float32([[100, 100], [160, 100], [160, 150], [100, 150]])
    assert ds.quad_from_lines([ds.line_through(p, q) for p, q in zip(tiny, np.roll(tiny, -1, axis=0))], (H, W)) is None, "меньше 10 %% кадра"
    rect = np.float32([[100, 80], [540, 80], [540, 400], [100, 400]])                                # стороны строго параллельны
    q = ds.quad_from_lines([ds.line_through(p, q) for p, q in zip(rect, np.roll(rect, -1, axis=0))], (H, W))
    assert q is not None and corners_close(q, rect, 0.5)


# ------------------------------------------------------- 4. геометрия ---
def test_order_corners_any_order_any_rotation():
    for perm in itertools.permutations(range(4)):
        assert np.allclose(ds.order_corners(QUAD[list(perm)]), QUAD), "все 24 порядка → tl, tr, br, bl"
    kite = np.float32([[330, 40], [560, 310], [360, 560], [150, 290]])       # верхний, правый, нижний, левый
    assert np.allclose(ds.order_corners(kite[[3, 1, 0, 2]]), kite), "первым идёт угол с минимальной x + y (здесь верхний), дальше по часовой"
    sheet =np.float32([[0, 0], [200, 0], [200, 283], [0, 283]]) - [100, 141.5]
    rng = np.random.default_rng(5)
    for deg in range(0, 360, 15):
        R = cv2.getRotationMatrix2D((0, 0), deg, 1.0)[:, :2]
        true = (sheet @ R.T + [320, 240]).astype(np.float32)                 # циклический порядок известен
        got = ds.order_corners(true[rng.permutation(4)])
        assert got.dtype == np.float32 and got.shape == (4, 2)
        assert any(np.allclose(got, np.roll(true, -k, axis=0), atol=1e-3) for k in range(4)), \
            "лист повёрнут на %d°: порядок не циклический — по таким углам гомография даёт «бабочку»" % deg


def test_rectify_size_and_content():
    scene = desk_scene(noise=0.0)
    front, Hm = ds.rectify(scene, QUAD[[2, 0, 3, 1]], size=(300, 424))
    assert front.shape[:2] == (424, 300) and np.asarray(Hm).shape == (3, 3)
    ink = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) < 110          # noqa: E731
    dark = cv2.cvtColor(front, cv2.COLOR_BGR2GRAY) < 0.55 * np.median(cv2.cvtColor(front, cv2.COLOR_BGR2GRAY))
    assert 100 * (dark == ink(page_img())).mean() > 98, "выпрямленный лист совпадает со страницей «в лоб»"
    f_a4, _ = ds.rectify(scene, QUAD, aspect=210 / 297)
    assert abs(f_a4.shape[1] / f_a4.shape[0] - 210 / 297) < 0.01 and abs(f_a4.shape[1] - 371) <= 1, "ширина — по самой длинной из сторон «верх / низ»: 371 px"
    side =np.float32([[300, 100], [420, 140], [420, 420], [300, 470]])    # съёмка сбоку: верх 126 px, левая сторона 370 px
    f_side, _ = ds.rectify(scene, side, aspect=210 / 297)
    assert f_side.shape[0] >= 368, "ни одна сторона не уменьшается: высота выхода не меньше длинной стороны в кадре (370 px), сейчас %d" % f_side.shape[0]
    assert abs(f_side.shape[1] / f_side.shape[0] - 210 / 297) < 0.01
    f_px, _ = ds.rectify(scene, QUAD)
    assert abs(f_px.shape[1] - 371) <= 1 and abs(f_px.shape[0] - 345) <= 1, "без size и aspect — по длинам сторон в кадре"


def test_quad_iou():
    a = np.float32([[100, 100], [300, 100], [300, 300], [100, 300]])
    b = a + [100, 0]
    assert ds.quad_iou(a, a, (H, W)) == pytest.approx(1.0)
    assert ds.quad_iou(a, b, (H, W)) == pytest.approx(1 / 3, abs=0.01), "половина перекрытия: IoU = 1/3"
    assert ds.quad_iou(a, a + [250, 0], (H, W)) == 0.0
    assert ds.quad_iou(a[[2, 0, 3, 1]], b, (H, W)) == pytest.approx(ds.quad_iou(a, b, (H, W))), "порядок углов не важен"
    assert ds.quad_iou(None, a, (H, W)) == 0.0 and ds.quad_iou(a, None, (H, W)) == 0.0
    assert 0.97 < ds.quad_iou(QUAD, QUAD + 1.0, (H, W)) < 1.0, "сдвиг на пиксель — IoU чуть меньше единицы"


# ------------------------------------------------ 5. тени и бинаризация ---
def shadow_page():
    page = cv2.cvtColor(page_img(), cv2.COLOR_BGR2GRAY)
    gt = page < 128
    yy, xx = np.mgrid[:424, :300].astype(np.float32)
    light = 1.0 - 0.55 * xx / 300                                          # свет падает слева
    m = np.zeros((424, 300), np.float32); cv2.fillPoly(m, [np.int32([[0, 150], [300, 60], [300, 170], [0, 300]])], 1.0)
    light *= 1 - 0.4 * cv2.GaussianBlur(m, (0, 0), 12)                     # и мягкая тень наискосок
    lit = page.astype(np.float32) * light + np.random.default_rng(6).normal(0, 2.0, page.shape)
    return np.clip(lit, 0, 255).astype(np.uint8), gt


def test_remove_shadows_flattens_light():
    lit, gt = shadow_page()
    flat = ds.remove_shadows(lit)
    assert flat.shape == lit.shape and flat.dtype == np.uint8
    paper = ~cv2.dilate(gt.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    assert lit[paper].std() > 25, "контроль: до выравнивания бумага неоднородна"
    assert flat[paper].std() < 8 and np.percentile(flat[paper], 5) > 225, "после деления на фон бумага ровная и светлая"
    assert flat[gt].mean() < 150, "чернила остались тёмными"
    otsu = lambda im: cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] < 128     # noqa: E731
    assert 100 * (otsu(lit) == gt).mean() < 90, "контроль: Otsu по неровному свету принимает тень за чернила"
    assert 100 * (otsu(flat) == gt).mean() > 99, "Otsu после выравнивания — больше 99 %% верных пикселей"
    assert 100 * ((ds.binarize(lit, "flat") < 128) == gt).mean() > 99


# ------------------------------------------------------------ пайплайн ---
def test_detect_both_paths_find_the_sheet():
    scene = desk_scene()
    for method in ("contour", "hough", "auto"):
        quad, info = ds.detect(scene, method)
        assert quad is not None, "путь %s лист не нашёл (края: %.2f %%, пороги %s)" % (method, info["edge_pct"], info["canny"])
        iou = ds.quad_iou(quad, QUAD, scene.shape)
        assert iou > 0.95, "путь %s: IoU %.3f" % (method, iou)
        assert np.allclose(quad, ds.order_corners(quad)), "углы возвращаются упорядоченными"
    assert info["path"] == "contour", "auto сначала пробует контуры"


def test_detect_covered_corner_and_empty_desk():
    covered = desk_scene(cover=(492, 408))                                  # кружка на правом нижнем углу листа
    quad, info = ds.detect(covered, "auto")
    assert quad is not None, "угол закрыт: контур не замкнут, но четыре стороны видны — лист находится по прямым"
    assert ds.quad_iou(quad, QUAD, covered.shape) > 0.93, "закрытый угол восстанавливается пересечением прямых"
    empty = desk_scene(sheet=False)
    assert ds.detect(empty, "auto")[0] is None, "листа нет — ответ None, а не четырёхугольник из волокон стола"


def test_run_video_counts_frames():
    frames = [desk_scene(seed=s) for s in range(4)] + [desk_scene(sheet=False, seed=9)]
    stats = ds.run_video(frames, "auto", seconds=0, show=False)
    assert stats["frames"] == 5 and stats["fps"] > 0
    assert stats["found_pct"] == pytest.approx(80.0), "лист есть в четырёх кадрах из пяти"


# ------------------------------------------------------------ CLI ---
def test_cli_detect_runs(tmp_path):
    src = tmp_path / "scene.png"; dst = tmp_path / "out.jpg"; ann = tmp_path / "corners.json"
    cv2.imwrite(str(src), desk_scene())
    ann.write_text(json.dumps({"scene.png": QUAD.tolist()}), encoding="utf-8")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "docscan.py"), "detect", str(src), "--ann", str(ann), "-o", str(dst)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr[-400:]
    rep = json.loads(r.stdout.strip().splitlines()[-1])
    assert rep["command"] == "detect" and rep["iou"] > 0.95 and dst.exists()
