import math
import numpy as np

from chart_common import *


def main():
    train, _, feature_cols = load_data()
    arrays = class_arrays(train, feature_cols)
    positions = np.arange(1, len(feature_cols) + 1)

    stats = {}
    lows, highs = [], []
    for label in LABELS:
        mean = arrays[label].mean(axis=0)
        sd = arrays[label].std(axis=0, ddof=1)
        stats[label] = (mean, sd)
        lows.append(mean - sd)
        highs.append(mean + sd)

    y_min = 0.0
    y_max = math.ceil((float(np.max(highs)) + 0.25) / 2) * 2

    width, height = 1180, 720
    x0, y0, w, h = 92, 112, 1010, 470
    x_scale = linear_scale(1, 155, x0, x0 + w)
    y_scale = linear_scale(y_min, y_max, y0 + h, y0)
    body = [chart_title(width, "三类工件 155 个检测位置平均曲线",
                        "均值曲线展示 A > B > C 的总体水平差异；阴影为各检测位置 ±1 标准差")]
    body.append(draw_axes(
        x0, y0, w, h,
        x_ticks=[1, 20, 40, 60, 80, 100, 120, 140, 155],
        y_ticks=np.arange(y_min, y_max + 0.1, 2),
        x_scale=x_scale,
        y_scale=y_scale,
        x_fmt=lambda v: str(int(v)),
        y_fmt=lambda v: fmt(v, 1),
    ))

    for label in LABELS:
        mean, sd = stats[label]
        upper = [(x_scale(x), y_scale(m + s)) for x, m, s in zip(positions, mean, sd)]
        lower = [(x_scale(x), y_scale(m - s)) for x, m, s in zip(positions[::-1], mean[::-1], sd[::-1])]
        body.append(polygon(upper + lower, COLORS[label], opacity=0.12))
    for label in LABELS:
        mean, _ = stats[label]
        pts = [(x_scale(x), y_scale(v)) for x, v in zip(positions, mean)]
        body.append(polyline(pts, COLORS[label], width=3.0))

    body.append(legend(x0 + 740, y0 - 16, [("A 类", COLORS["A"]), ("B 类", COLORS["B"]), ("C 类", COLORS["C"])], gap=80))
    body.append(text(x0 + w / 2, y0 + h + 56, "检测位置", size=13, fill="#37474F", anchor="middle"))
    body.append(
        f'<text x="{x0 - 60}" y="{y0 + h / 2}" font-family="{FONT}" font-size="13" '
        f'fill="#37474F" text-anchor="middle" transform="rotate(-90 {x0 - 60},{y0 + h / 2})">X 光测量值</text>'
    )
    save_svg(CHART_DIR / "01_mean_curves.svg", width, height, "\n".join(body))


if __name__ == "__main__":
    main()
