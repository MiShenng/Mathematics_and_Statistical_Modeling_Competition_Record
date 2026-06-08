import pandas as pd

from chart_common import *


def main():
    df = pd.read_csv(ROOT / "outputs" / "final_feature_importance.csv", encoding="utf-8-sig").head(15)
    df = df.iloc[::-1].reset_index(drop=True)

    width, height = 1040, 760
    x0, y0, w, h = 188, 110, 780, 520
    x_min, x_max = 0.50, 0.55
    x_scale = linear_scale(x_min, x_max, x0, x0 + w)
    row_h = h / len(df)

    body = [chart_title(width, "关键检测位置区分力排名（eta²）",
                        "eta² 表示类别差异解释该位置总变异的比例；前 15 个位置均超过 0.52")]
    body.append(rect(x0, y0, w, h, fill="#FFFFFF", stroke="#D8DEE9", sw=1, rx=8))
    for xt in [0.50, 0.51, 0.52, 0.53, 0.54, 0.55]:
        x = x_scale(xt)
        body.append(line(x, y0, x, y0 + h, stroke=COLORS["grid"], width=1))
        body.append(text(x, y0 + h + 22, f"{xt:.2f}", size=11, fill="#60717F", anchor="middle"))

    for i, row in df.iterrows():
        cy = y0 + i * row_h + row_h / 2
        eta = float(row["eta2"])
        bar_x = x_scale(x_min)
        bar_w = x_scale(eta) - bar_x
        color = lerp_color("#9EC5E5", "#2A6FBB", i / max(len(df) - 1, 1))
        body.append(text(x0 - 14, cy + 5, f"位置 {int(row['检测位置'])}", size=12, fill="#37474F", anchor="end"))
        body.append(rect(bar_x, cy - row_h * 0.31, bar_w, row_h * 0.62, fill=color, rx=5, opacity=0.95))
        body.append(text(bar_x + bar_w + 8, cy + 5, fmt(eta, 4), size=11, fill="#37474F"))

    body.append(text(x0 + w / 2, y0 + h + 54, "eta²", size=13, fill="#37474F", anchor="middle"))
    body.append(text(x0 + w - 120, y0 - 18, "区分力越强 →", size=12, fill="#60717F"))
    save_svg(CHART_DIR / "04_feature_importance.svg", width, height, "\n".join(body))


if __name__ == "__main__":
    main()
