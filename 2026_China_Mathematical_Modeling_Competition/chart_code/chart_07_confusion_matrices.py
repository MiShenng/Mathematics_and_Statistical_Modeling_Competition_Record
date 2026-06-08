import json

from chart_common import *


def draw_matrix(panel_x, panel_y, title, cm, color):
    cell = 82
    matrix_y = panel_y + 34
    parts = []
    parts.append(text(panel_x + cell * 1.5, panel_y - 26, title, size=16, weight=700, anchor="middle"))
    parts.append(text(panel_x + cell * 1.5, panel_y - 6, "预测类别", size=11, fill="#60717F", anchor="middle"))
    for j, lab in enumerate(LABELS):
        parts.append(text(panel_x + (j + 0.5) * cell, panel_y + 16, lab, size=12, weight=700, fill="#37474F", anchor="middle"))
    for i, lab in enumerate(LABELS):
        parts.append(text(panel_x - 14, matrix_y + (i + 0.5) * cell + 4, lab, size=12, weight=700, fill="#37474F", anchor="end"))
    parts.append(text(panel_x - 54, matrix_y + cell * 1.5, "真实类别", size=11, fill="#60717F", anchor="middle",
                      extra=f'transform="rotate(-90 {panel_x - 54},{matrix_y + cell * 1.5})"'))

    for i in range(3):
        row_sum = sum(cm[i])
        for j in range(3):
            v = cm[i][j]
            rate = v / row_sum if row_sum else 0
            fill = lerp_color("#F3F6FA", color, rate)
            x = panel_x + j * cell
            y = matrix_y + i * cell
            parts.append(rect(x, y, cell, cell, fill=fill, stroke="#FFFFFF", sw=2, rx=6))
            txt_color = "#FFFFFF" if rate > 0.65 else "#263238"
            parts.append(text(x + cell / 2, y + cell / 2 - 2, str(v), size=17, weight=700, fill=txt_color, anchor="middle"))
            parts.append(text(x + cell / 2, y + cell / 2 + 18, pct(rate, 1), size=10, fill=txt_color, anchor="middle"))
    return "\n".join(parts)


def main():
    data = json.loads((ROOT / "outputs" / "final_results.json").read_text(encoding="utf-8"))
    result_map = {r["model"]: r for r in data["cv_results"]}
    panels = [
        ("高斯朴素贝叶斯", result_map["高斯朴素贝叶斯"]["confusion_matrix"], "#4E79A7"),
        ("正则化 LDA", result_map["正则化LDA"]["confusion_matrix"], "#F28E2B"),
        ("正则化 QDA", result_map["正则化QDA"]["confusion_matrix"], "#6E4AA8"),
    ]

    width, height = 1180, 560
    body = [chart_title(width, "主要分类模型混淆矩阵对比",
                        "单元格显示样本数与行内比例；QDA 在三类上均实现完全分类")]
    start_x, y = 110, 138
    gap = 350
    for k, (title, cm, color) in enumerate(panels):
        body.append(draw_matrix(start_x + k * gap, y, title, cm, color))
    save_svg(CHART_DIR / "07_confusion_matrices.svg", width, height, "\n".join(body))


if __name__ == "__main__":
    main()
