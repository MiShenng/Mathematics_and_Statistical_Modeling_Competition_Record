import pandas as pd

from chart_common import *


def main():
    df = pd.read_csv(ROOT / "outputs" / "final_model_metrics.csv", encoding="utf-8-sig")
    metrics = [("总准确率", "#4E79A7"), ("平衡准确率", "#F28E2B"), ("宏F1", "#59A14F")]

    width, height = 1160, 720
    x0, y0, w, h = 86, 112, 980, 470
    y_min, y_max = 0.90, 1.005
    y_scale = linear_scale(y_min, y_max, y0 + h, y0)

    body = [chart_title(width, "五类非神经网络模型性能对比",
                        "QDA 达到满分；NB 与 PCA-QDA 表现稳定，最近质心/LDA 主要在 A/B 边界上损失")]
    body.append(rect(x0, y0, w, h, fill="#FFFFFF", stroke="none", sw=0, rx=0))
    for yt in [0.90, 0.925, 0.95, 0.975, 1.00]:
        yy = y_scale(yt)
        body.append(line(x0, yy, x0 + w, yy, stroke=COLORS["grid"], width=1))
        body.append(text(x0 - 10, yy + 4, pct(yt, 1), size=11, fill="#60717F", anchor="end"))
    body.append(line(x0, y0 + h, x0 + w, y0 + h, stroke="#263238", width=1.2))
    body.append(line(x0, y0, x0, y0 + h, stroke="#263238", width=1.2))

    n = len(df)
    group_w = w / n
    bar_w = group_w * 0.18
    for i, row in df.iterrows():
        cx = x0 + group_w * (i + 0.5)
        for j, (metric, color) in enumerate(metrics):
            val = float(row[metric])
            x = cx + (j - 1) * bar_w * 1.25 - bar_w / 2
            yy = y_scale(val)
            body.append(rect(x, yy, bar_w, y0 + h - yy, fill=color, rx=3, opacity=0.92))
            if val >= 0.995:
                label_y = yy - 8
            else:
                label_y = yy - 7
            body.append(text(x + bar_w / 2, label_y, fmt(val, 3), size=10, fill="#37474F", anchor="middle"))
        body.append(text(cx, y0 + h + 22, row["模型"], size=11, fill="#37474F", anchor="middle"))

    body.append(legend(x0 + 610, y0 - 16, metrics, gap=108))
    body.append(text(x0 + w / 2, y0 + h + 58, "模型", size=13, fill="#37474F", anchor="middle"))
    body.append(text(x0 - 58, y0 + h / 2, "指标值", size=13, fill="#37474F", anchor="middle",
                     extra=f'transform="rotate(-90 {x0 - 58},{y0 + h / 2})"'))
    save_svg(CHART_DIR / "06_model_performance.svg", width, height, "\n".join(body))


if __name__ == "__main__":
    main()
