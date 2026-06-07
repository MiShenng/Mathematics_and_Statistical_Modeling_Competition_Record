import numpy as np

from chart_common import *


def main():
    train, _, feature_cols = load_data()
    X = train[feature_cols].to_numpy(float)
    y = train["label"].to_numpy()
    scores, ratio = pca_scores(X)

    rng = np.random.default_rng(20260605)
    sample_idx = []
    for label in LABELS:
        idx = np.where(y == label)[0]
        take = min(len(idx), 1200)
        sample_idx.extend(rng.choice(idx, size=take, replace=False).tolist())
    sample_idx = np.array(sample_idx)

    x_all = scores[:, 0]
    y_all = scores[:, 1]
    x_pad = (x_all.max() - x_all.min()) * 0.06
    y_pad = (y_all.max() - y_all.min()) * 0.10
    x_min, x_max = float(x_all.min() - x_pad), float(x_all.max() + x_pad)
    y_min, y_max = float(y_all.min() - y_pad), float(y_all.max() + y_pad)

    width, height = 1080, 760
    x0, y0, w, h = 92, 112, 860, 510
    x_scale = linear_scale(x_min, x_max, x0, x0 + w)
    y_scale = linear_scale(y_min, y_max, y0 + h, y0)

    body = [chart_title(width, "PCA 主成分空间中的三类工件分布",
                        f"PC1 解释 {pct(ratio[0], 2)} 方差；散点为分层抽样，类中心由全部样本计算")]
    body.append(draw_axes(
        x0, y0, w, h,
        x_ticks=nice_ticks(x_min, x_max, 7),
        y_ticks=nice_ticks(y_min, y_max, 7),
        x_scale=x_scale,
        y_scale=y_scale,
        x_fmt=lambda v: fmt(v, 0),
        y_fmt=lambda v: fmt(v, 1),
    ))

    for label in ["C", "B", "A"]:
        idx = sample_idx[y[sample_idx] == label]
        for i in idx:
            body.append(circle(x_scale(scores[i, 0]), y_scale(scores[i, 1]), 2.1, COLORS[label], opacity=0.22))

    for label in LABELS:
        center = scores[y == label, :2].mean(axis=0)
        cx, cy = x_scale(center[0]), y_scale(center[1])
        body.append(circle(cx, cy, 8, "#FFFFFF", stroke=COLORS[label], sw=3, opacity=1))
        body.append(text(cx + 12, cy + 4, f"{label} 类中心", size=12, fill=COLORS[label], weight=700))

    body.append(legend(x0 + 640, y0 - 16, [("A 类", COLORS["A"]), ("B 类", COLORS["B"]), ("C 类", COLORS["C"])], gap=80))
    body.append(text(x0 + w / 2, y0 + h + 56, f"PC1 ({pct(ratio[0], 2)})", size=13, fill="#37474F", anchor="middle"))
    body.append(
        f'<text x="{x0 - 58}" y="{y0 + h / 2}" font-family="{FONT}" font-size="13" '
        f'fill="#37474F" text-anchor="middle" transform="rotate(-90 {x0 - 58},{y0 + h / 2})">PC2 ({pct(ratio[1], 2)})</text>'
    )
    save_svg(CHART_DIR / "02_pca_scatter.svg", width, height, "\n".join(body))


if __name__ == "__main__":
    main()
