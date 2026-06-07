import numpy as np

from chart_common import *


def main():
    train, _, feature_cols = load_data()
    X = train[feature_cols].to_numpy(float)
    _, ratio = pca_scores(X)
    n = 20
    xs = np.arange(1, n + 1)
    r = ratio[:n]
    cum = np.cumsum(r)

    width, height = 1080, 720
    x0, y0, w, h = 92, 112, 860, 470
    x_scale = linear_scale(0.4, n + 0.6, x0, x0 + w)
    y_scale = linear_scale(0, 0.58, y0 + h, y0)
    y2_scale = linear_scale(0, 0.70, y0 + h, y0)

    body = [chart_title(width, "PCA 碎石图与累计解释率",
                        "PC1 解释 54.17% 方差，PC2 以后单个主成分贡献迅速下降")]
    body.append(draw_axes(
        x0, y0, w, h,
        x_ticks=[1, 5, 10, 15, 20],
        y_ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5],
        x_scale=x_scale,
        y_scale=y_scale,
        x_fmt=lambda v: f"PC{int(v)}",
        y_fmt=lambda v: pct(v, 0),
    ))

    bar_w = w / n * 0.58
    for pc, val in zip(xs, r):
        x = x_scale(pc) - bar_w / 2
        y = y_scale(val)
        body.append(rect(x, y, bar_w, y0 + h - y, fill="#5A8CC8", rx=3, opacity=0.88))
    line_pts = [(x_scale(pc), y2_scale(v)) for pc, v in zip(xs, cum)]
    body.append(polyline(line_pts, "#D98C1F", width=3.0))
    for pc, val in zip(xs, cum):
        if pc in [1, 5, 10, 15, 20]:
            body.append(circle(x_scale(pc), y2_scale(val), 4, "#D98C1F", stroke="#FFFFFF", sw=1.5))

    # Right axis for cumulative ratio.
    body.append(line(x0 + w, y0, x0 + w, y0 + h, stroke="#263238", width=1.2))
    for yt in [0, 0.2, 0.4, 0.6]:
        yy = y2_scale(yt)
        body.append(text(x0 + w + 10, yy + 4, pct(yt, 0), size=11, fill="#60717F"))

    body.append(legend(x0 + 590, y0 - 16, [("单个解释率", "#5A8CC8"), ("累计解释率", "#D98C1F")], gap=118))
    body.append(text(x0 + w / 2, y0 + h + 56, "主成分", size=13, fill="#37474F", anchor="middle"))
    body.append(text(x0 - 58, y0 + h / 2, "单个解释率", size=13, fill="#37474F", anchor="middle",
                     extra=f'transform="rotate(-90 {x0 - 58},{y0 + h / 2})"'))
    body.append(text(x0 + w + 62, y0 + h / 2, "累计解释率", size=13, fill="#37474F", anchor="middle",
                     extra=f'transform="rotate(90 {x0 + w + 62},{y0 + h / 2})"'))
    body.append(text(x_scale(1), y_scale(r[0]) - 10, pct(r[0], 2), size=12, fill="#2A6FBB", anchor="middle", weight=700))
    save_svg(CHART_DIR / "03_pca_scree.svg", width, height, "\n".join(body))


if __name__ == "__main__":
    main()
