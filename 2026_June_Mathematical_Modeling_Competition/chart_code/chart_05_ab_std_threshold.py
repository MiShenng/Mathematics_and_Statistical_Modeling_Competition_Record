import numpy as np

from chart_common import *


def main():
    train, _, feature_cols = load_data()
    arrays = class_arrays(train, feature_cols)
    sd_a = arrays["A"].std(axis=1, ddof=1)
    sd_b = arrays["B"].std(axis=1, ddof=1)
    threshold = 1.1058551798393363

    x_min = float(min(sd_a.min(), sd_b.min()) - 0.06)
    x_max = float(max(sd_a.max(), sd_b.max()) + 0.06)
    bins = np.linspace(x_min, x_max, 54)
    ha, edges = np.histogram(sd_a, bins=bins, density=True)
    hb, _ = np.histogram(sd_b, bins=bins, density=True)
    y_max = float(max(ha.max(), hb.max()) * 1.16)

    width, height = 1080, 720
    x0, y0, w, h = 92, 112, 870, 470
    x_scale = linear_scale(x_min, x_max, x0, x0 + w)
    y_scale = linear_scale(0, y_max, y0 + h, y0)

    y_ticks = [0, 1, 2, 3, 4, 5]
    x_ticks = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

    body = [chart_title(width, "A/B 类样本内标准差分布与最优阈值",
                        "直方图高度为归一化密度  A 类全部高于阈值  B 类全部低于阈值")]
    body.append(rect(x0, y0, w, h, fill="#FFFFFF", stroke="none", sw=0, rx=0))
    for yt in y_ticks:
        yy = y_scale(yt)
        body.append(line(x0, yy, x0 + w, yy, stroke=COLORS["grid"], width=1))
        body.append(text(x0 - 10, yy + 4, fmt(yt, 1), size=11, fill="#60717F", anchor="end"))
    for xt in x_ticks:
        xx = x_scale(xt)
        body.append(line(xx, y0 + h, xx, y0 + h + 5, stroke="#60717F", width=1))
        body.append(text(xx, y0 + h + 22, fmt(xt, 2), size=11, fill="#60717F", anchor="middle"))
    body.append(line(x0, y0 + h, x0 + w, y0 + h, stroke="#263238", width=1.2))
    body.append(line(x0, y0, x0, y0 + h, stroke="#263238", width=1.2))

    for hist, color, shift in [(hb, COLORS["B"], 0), (ha, COLORS["A"], 0)]:
        for count, left, right in zip(hist, edges[:-1], edges[1:]):
            x = x_scale(left)
            ww = max(0.5, x_scale(right) - x_scale(left) - 1)
            yy = y_scale(count)
            body.append(rect(x, yy, ww, y0 + h - yy, fill=color, opacity=0.34, rx=1))

    tx = x_scale(threshold)
    body.append(line(tx, y0, tx, y0 + h, stroke="#B71C1C", width=2.5, dash="6 5"))
    body.append(text(tx + 10, y0 + 24, "阈值 s* = 1.105855", size=13, fill="#B71C1C", weight=700))
    body.append(text(x_scale(sd_b.max()), y0 + h - 18, f"B 最大 {fmt(sd_b.max(), 6)}", size=12, fill=COLORS["B"], anchor="end"))
    body.append(text(x_scale(sd_a.min()), y0 + h - 18, f"A 最小 {fmt(sd_a.min(), 6)}", size=12, fill=COLORS["A"]))

    body.append(legend(x0 + 650, y0 - 16, [("A 类", COLORS["A"]), ("B 类", COLORS["B"])], gap=80))
    body.append(text(x0 + w / 2, y0 + h + 56, "单个工件 155 个检测值的样本内标准差", size=13, fill="#37474F", anchor="middle"))
    body.append(text(x0 - 58, y0 + h / 2, "归一化密度", size=13, fill="#37474F", anchor="middle",
                     extra=f'transform="rotate(-90 {x0 - 58},{y0 + h / 2})"'))
    save_svg(CHART_DIR / "05_ab_std_threshold.svg", width, height, "\n".join(body))


if __name__ == "__main__":
    main()
