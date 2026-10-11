#!/usr/bin/env bash
# ДЗ 1 «Фотолаборатория» — все команды одним прогоном.
#
#   ./run_all.sh                 свои фото из data/, результаты в out/
#   ./run_all.sh data out        то же, пути явно
#   ./run_all.sh --course        кадры курса: числа для сверки с sample_data/README.md
#
# Каждая команда печатает строку JSON; весь вывод дублируется в <out>/log.jsonl —
# из него заполняются таблицы REPORT.md. Переопределяется: PHOTOLAB=…, PYTHON=….
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PL="${PHOTOLAB:-$HERE/photolab.py}"
PY="${PYTHON:-python3}"

run() {                                   # печатает команду, её JSON и всё это — в лог
    printf '\n+ %s\n' "$*" | tee -a "$LOG"
    "$@" | tee -a "$LOG"
}

# ------------------------------------------------- кадры курса: сверка эталона ---
if [ "${1:-}" = "--course" ]; then
    REPO="$(cd "$HERE/../../.." && pwd)"
    IU="$REPO/lectures/L05-filtering/demo/iu-building.jpg"
    L3="$REPO/lectures/L03-color-histograms/demo"
    L4="$REPO/lectures/L04-geometry/demo"
    OUT="${2:-out-course}"; LOG="$OUT/log.jsonl"
    mkdir -p "$OUT"; : > "$LOG"
    echo "кадры курса → $OUT  (эталонные числа: homeworks/hw1-photolab/sample_data/README.md)"

    run "$PY" "$PL" gamma "$IU" --gamma 0.5 -o "$OUT/g05.jpg" --hist "$OUT/g05_hist.png"
    run "$PY" "$PL" gamma "$IU" --gamma 2.0 -o "$OUT/g20.jpg"
    run "$PY" "$PL" wb "$L3/ball-lamp.jpg" --method grey-world  -o "$OUT/wb_gw.jpg"
    run "$PY" "$PL" wb "$L3/ball-lamp.jpg" --method white-patch -o "$OUT/wb_wp.jpg"
    run "$PY" "$PL" clahe "$IU" --clip 2 --tiles 8 -o "$OUT/clahe.jpg" --compare "$OUT/clahe_cmp.jpg"
    # тот же CLAHE на равномерно недоэкспонированном кадре — оговорка раздела 2.3 отчёта
    run "$PY" "$PL" clahe "$L3/ball-lamp.jpg" --clip 2 --tiles 8 -o "$OUT/clahe_lamp.jpg" \
        --compare "$OUT/clahe_lamp_cmp.jpg"
    run "$PY" "$PL" chromakey "$L3/ball-lamp.jpg" --bg "$L4/road-cam.jpg" --lo 140,121,40 --hi 168,230,255 \
        -o "$OUT/ck.jpg" --mask "$OUT/ck_mask.png"
    run "$PY" "$PL" chromakey "$L3/ball-day.jpg" --bg "$L4/road-cam.jpg" --sample 160,530,60,60 \
        -o "$OUT/ck2.jpg" --mask "$OUT/ck2_mask.png"
    run "$PY" "$PL" rotate "$IU" --deg 30 -o "$OUT/rot30.jpg"
    run "$PY" "$PL" rotate "$IU" --deg -137 -o "$OUT/rotm137.jpg"      # контроль знака: холст по формуле и при отрицательном угле
    run "$PY" "$PL" resize "$IU" --scale 0.25 --report -o "$OUT/small.jpg"
    run "$PY" "$PL" resize "$IU" --scale 2 -o "$OUT/big.jpg"
    run "$PY" "$PL" rectify "$L4/poster-angle.jpg" --corners 1290,1010,560,330,600,840,1250,220 --aspect 0.707 \
        -o "$OUT/front.jpg"
    run "$PY" "$PL" rectify "$L4/poster-angle.jpg" --corners 560,330,1250,220,1290,1010,600,840 --size 640x905 \
        -o "$OUT/front640.jpg"
    run "$PY" "$PL" sheet "$IU" "$L3/ball-day.jpg" "$L4/poster-angle.jpg" "$L4/road-cam.jpg" "$L4/text-page.png" \
        --cols 3 -o "$OUT/sheet.jpg"

    echo; echo "--- числа, которые команды не печатают (metrics.py) ---" | tee -a "$LOG"
    run "$PY" "$HERE/metrics.py" recall "$OUT/ck_mask.png" --circle 190,560,58
    run "$PY" "$HERE/metrics.py" recall "$OUT/ck2_mask.png" --circle 190,560,58
    run "$PY" "$HERE/metrics.py" match "$OUT/front640.jpg" "$L4/poster-gt.png"

    # вопрос 2.1 отчёта: одни и те же границы с одного прямоугольника — по HSV и прямо по B, G, R
    # сверка с числами лекций: L3 демо 1.4 (стр. 30) и L4 стр. 19 / демо 2.4 (стр. 67)
    echo; echo "--- сверка с лекциями ---" | tee -a "$LOG"
    run "$PY" "$HERE/experiments.py" lecture-check "$IU"

    echo; echo "--- опыт раздела 2.1: HSV против RGB ---" | tee -a "$LOG"
    run "$PY" "$HERE/experiments.py" rgb-vs-hsv --day "$L3/ball-day.jpg" --lamp "$L3/ball-lamp.jpg"

    # раздел 3 «где сломалось»: оттенок мяча сдвинут на разрыв 0/179 — красного предмета на кадрах курса нет
    echo; echo "--- опыт раздела 3: разрыв оттенка 0/179 ---" | tee -a "$LOG"
    run "$PY" "$HERE/experiments.py" hue-shift "$L3/ball-day.jpg" --shift 27 -o "$OUT/ball-red.png"
    run "$PY" "$PL" chromakey "$OUT/ball-red.png" --bg "$L4/road-cam.jpg" --sample 160,530,60,60 \
        -o "$OUT/ck_red_auto.jpg" --mask "$OUT/ck_red_auto_mask.png"
    run "$PY" "$HERE/metrics.py" recall "$OUT/ck_red_auto_mask.png" --circle 190,560,58
    run "$PY" "$PL" chromakey "$OUT/ball-red.png" --bg "$L4/road-cam.jpg" --lo 167,100,40 --hi 13,255,255 \
        -o "$OUT/ck_red_manual.jpg" --mask "$OUT/ck_red_manual_mask.png"
    run "$PY" "$HERE/metrics.py" recall "$OUT/ck_red_manual_mask.png" --circle 190,560,58
    echo; echo "готово. лог: $LOG"
    exit 0
fi

# ------------------------------------------------------------- свои фотографии ---
DATA="${1:-data}"; OUT="${2:-out}"; LOG="$OUT/log.jsonl"

missing=""
for f in dark.jpg bright.jpg object.jpg bg.jpg poster.jpg text.jpg; do
    [ -f "$DATA/$f" ] || missing="$missing $f"
done
if [ -n "$missing" ]; then
    echo "в $DATA/ не хватает файлов:$missing" >&2
    echo "что снимать — README.md задания, §2 (HEIC не читается: sips -s format jpeg in.heic --out $DATA/dark.jpg)" >&2
    exit 1
fi

mkdir -p "$OUT"; : > "$LOG"

# прямоугольник на объекте для chromakey --sample: из файла или центр кадра
if [ -f "$DATA/object-sample.txt" ]; then
    SAMPLE="$(tr -d ' \n' < "$DATA/object-sample.txt")"
else
    SAMPLE="$("$PY" - "$DATA/object.jpg" <<'PYEOF'
import sys, cv2
h, w = cv2.imread(sys.argv[1]).shape[:2]
print("%d,%d,%d,%d" % (w * 9 // 20, h * 9 // 20, w // 10, h // 10))
PYEOF
)"
    echo "! --sample не задан: беру центральные 10 % кадра ($SAMPLE)."
    echo "  Если объект не в центре — положите свои x,y,w,h в $DATA/object-sample.txt"
fi

# реальные пропорции объекта W/H: A4 — 0.707, своё значение — в data/poster-aspect.txt
ASPECT="0.707"
[ -f "$DATA/poster-aspect.txt" ] && ASPECT="$(tr -d ' \n' < "$DATA/poster-aspect.txt")"

# четыре угла плаката: из файла или кликами в окне OpenCV
if [ -f "$DATA/poster-corners.txt" ]; then
    CORNERS="$(tr -d ' \n' < "$DATA/poster-corners.txt")"
    RECTIFY_ARGS=(--corners "$CORNERS")
else
    echo "! $DATA/poster-corners.txt не найден: углы задаются кликами (4 клика, Enter)."
    echo "  Числа из вывода стоит сохранить в $DATA/poster-corners.txt — тогда прогон станет воспроизводимым"
    RECTIFY_ARGS=(--clicks)
fi

run "$PY" "$PL" gamma "$DATA/dark.jpg"   --gamma 0.5 -o "$OUT/dark_g05.jpg" --hist "$OUT/dark_g05_hist.png"
run "$PY" "$PL" gamma "$DATA/dark.jpg"   --gamma 2.0 -o "$OUT/dark_g20.jpg" --hist "$OUT/dark_g20_hist.png"
run "$PY" "$PL" wb    "$DATA/bright.jpg" --method grey-world  -o "$OUT/bright_gw.jpg" --hist "$OUT/bright_gw_hist.png"
run "$PY" "$PL" wb    "$DATA/bright.jpg" --method white-patch -o "$OUT/bright_wp.jpg" --hist "$OUT/bright_wp_hist.png"
# что баланс белого сделал с белым столом, а не со средними по кадру — раздел 3 отчёта
run "$PY" "$HERE/experiments.py" wb-reference "$DATA/bright.jpg"
run "$PY" "$PL" clahe "$DATA/dark.jpg"   --clip 2 --tiles 8 -o "$OUT/dark_clahe.jpg" \
    --compare "$OUT/dark_cmp.jpg" --hist "$OUT/dark_clahe_hist.png"
run "$PY" "$PL" chromakey "$DATA/object.jpg" --bg "$DATA/bg.jpg" --sample "$SAMPLE" \
    -o "$OUT/object_on_bg.jpg" --mask "$OUT/object_mask.png"
run "$PY" "$PL" rotate "$DATA/poster.jpg" --deg 30   -o "$OUT/poster_rot30.jpg"
run "$PY" "$PL" rotate "$DATA/poster.jpg" --deg -137 -o "$OUT/poster_rotm137.jpg"
run "$PY" "$PL" resize "$DATA/text.jpg" --scale 0.25 --report -o "$OUT/text_x025.jpg"
run "$PY" "$PL" resize "$DATA/text.jpg" --scale 2 -o "$OUT/text_x2.jpg"
run "$PY" "$PL" rectify "$DATA/poster.jpg" "${RECTIFY_ARGS[@]}" --aspect "$ASPECT" -o "$OUT/poster_front.jpg"
# откуда взялся --aspect: пропорции из перспективы против наивных «по сторонам в кадре»
if [ -n "${CORNERS:-}" ]; then
    LONG_MM=()
    [ -f "$DATA/poster-long-mm.txt" ] && LONG_MM=(--long-mm "$(tr -d ' \n' < "$DATA/poster-long-mm.txt")")
    run "$PY" "$HERE/experiments.py" aspect "$DATA/poster.jpg" --corners "$CORNERS" "${LONG_MM[@]}"
fi
run "$PY" "$PL" sheet "$DATA"/*.jpg "$OUT/dark_g05.jpg" "$OUT/bright_gw.jpg" "$OUT/object_on_bg.jpg" \
    "$OUT/poster_front.jpg" --cols 4 -o "$OUT/sheet.jpg"

cat <<MSG

готово. всё в $OUT/, лог — $LOG

осталось для REPORT.md:
  1. таблицы 1.1–1.6 — числа из $LOG
  2. числа на кадрах курса (рубрика C требует и их):  ./run_all.sh --course
  3. три ответа и раздел «где сломалось» — своими словами, с числами из таблиц
MSG
