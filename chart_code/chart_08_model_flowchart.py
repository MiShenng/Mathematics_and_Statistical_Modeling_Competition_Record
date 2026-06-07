from chart_common import *


BLACK = "#111111"
GRAY = "#777777"
LIGHT = "#FFFFFF"


def connector(points, color=BLACK, width=1.5):
    parts = [polyline(points, stroke=color, width=width, opacity=1)]
    if len(points) < 2:
        return "\n".join(parts)

    x1, y1 = points[-2]
    x2, y2 = points[-1]
    dx, dy = x2 - x1, y2 - y1
    length = (dx * dx + dy * dy) ** 0.5 or 1
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 9
    base = (x2 - ux * size, y2 - uy * size)
    tip = (x2, y2)
    p1 = (base[0] + px * size * 0.5, base[1] + py * size * 0.5)
    p2 = (base[0] - px * size * 0.5, base[1] - py * size * 0.5)
    parts.append(polygon([tip, p1, p2], fill=color, opacity=1))
    return "\n".join(parts)


def multi_text(cx, cy, lines, size=17, leading=24, weight=400):
    total = leading * (len(lines) - 1)
    start_y = cy - total / 2 + size / 3
    return "\n".join(
        text(cx, start_y + i * leading, line_text, size=size, weight=weight, fill=BLACK, anchor="middle")
        for i, line_text in enumerate(lines)
    )


def process_box(cx, cy, w, h, lines, size=17, leading=24):
    x = cx - w / 2
    y = cy - h / 2
    parts = [
        rect(x, y, w, h, fill=LIGHT, stroke=BLACK, sw=1.4, rx=6),
        multi_text(cx, cy, lines, size=size, leading=leading),
    ]
    return "\n".join(parts)


def decision_box(cx, cy, w, h, lines, size=17, leading=24):
    points = [
        (cx, cy - h / 2),
        (cx + w / 2, cy),
        (cx, cy + h / 2),
        (cx - w / 2, cy),
    ]
    parts = [
        polygon(points, fill=LIGHT, opacity=1, stroke=BLACK, sw=1.4),
        multi_text(cx, cy, lines, size=size, leading=leading),
    ]
    return "\n".join(parts)


def small_label(x, y, content):
    return text(x, y, content, size=15, fill=BLACK, anchor="middle")


def branch_label(x, y, content):
    parts = [
        rect(x - 64, y - 20, 128, 26, fill=LIGHT, stroke="none", sw=0, rx=0),
        text(x, y, content, size=16, fill=BLACK, anchor="middle"),
    ]
    return "\n".join(parts)


def corner_marks(width, height):
    m = 44
    l = 34
    parts = [
        line(m, m, m, m + l, stroke=GRAY, width=0.8, opacity=0.75),
        line(m, m + l, m - l, m + l, stroke=GRAY, width=0.8, opacity=0.75),
        line(width - m, m, width - m, m + l, stroke=GRAY, width=0.8, opacity=0.75),
        line(width - m, m + l, width - m + l, m + l, stroke=GRAY, width=0.8, opacity=0.75),
    ]
    return "\n".join(parts)


def main():
    width, height = 1120, 1480
    cx = width / 2
    body = [corner_marks(width, height)]

    # Top main stream
    body.append(process_box(cx, 86, 300, 54, ["研究开始"], size=18))
    body.append(process_box(cx, 178, 420, 76, ["数据读取与变量构建", "A/B/C已知样本  D1-D10未知样本", "检测位置 X1-X155"], size=16, leading=22))
    body.append(process_box(cx, 292, 420, 78, ["数据预处理", "缺失检查  列一致性  异常值初筛", "训练折 Z-score 标准化"], size=16, leading=22))
    body.append(decision_box(cx, 418, 300, 92, ["数据质量", "是否合格？"], size=17, leading=25))
    body.append(process_box(248, 418, 250, 66, ["修正数据问题", "保留有效样本集"], size=16, leading=23))

    body.append(connector([(cx, 113), (cx, 140)]))
    body.append(connector([(cx, 216), (cx, 253)]))
    body.append(connector([(cx, 331), (cx, 372)]))
    body.append(connector([(410, 418), (373, 418)]))
    body.append(small_label(392, 404, "否"))
    body.append(connector([(248, 451), (248, 495), (cx, 495), (cx, 526)]))
    body.append(connector([(cx, 464), (cx, 526)]))
    body.append(small_label(588, 501, "是"))

    # Feature analysis and core evidence
    body.append(process_box(cx, 566, 420, 82, ["特征分析", "均值曲线  样本内标准差  ANOVA", "PCA解释率与类中心分离"], size=16, leading=22))
    body.append(process_box(cx, 685, 420, 76, ["关键判别依据形成", "A/B样本内SD阈值  重要检测位置", "PC1形态差异与先验选择"], size=16, leading=22))
    body.append(decision_box(cx, 818, 300, 92, ["后续分析目标", "解释机制 / 预测分类"], size=17, leading=25))

    body.append(connector([(cx, 607), (cx, 647)]))
    body.append(connector([(cx, 723), (cx, 772)]))

    # Two-column modeling stream
    left_x = 280
    right_x = 840
    branch_top = 864
    bw = 310
    bh = 72
    body.append(branch_label(392, branch_top - 12, "解释与稳健性"))
    body.append(branch_label(728, branch_top - 12, "预测建模"))
    body.append(connector([(cx - 150, 818), (left_x, 818), (left_x, branch_top)]))
    body.append(connector([(cx + 150, 818), (right_x, 818), (right_x, branch_top)]))

    left_nodes = [
        (branch_top + 36, ["基准模型诊断", "最近质心  NB  LDA"]),
        (branch_top + 140, ["QDA合理性检验", "A/B重叠  方差差异  逐折指标"]),
        (branch_top + 244, ["敏感性分析", "先验概率  正则化参数  PCA维数"]),
    ]
    right_nodes = [
        (branch_top + 36, ["主分类器构建", "正则化QDA  经验先验"]),
        (branch_top + 140, ["低维复核模型", "PCA-QDA 与维数扫描"]),
        (branch_top + 244, ["一致性判定", "5模型投票  少于4/5触发复检"]),
    ]

    for cy, lines in left_nodes:
        body.append(process_box(left_x, cy, bw, bh, lines, size=16, leading=23))
    for cy, lines in right_nodes:
        body.append(process_box(right_x, cy, bw, bh, lines, size=16, leading=23))

    for nodes_x in [left_x, right_x]:
        body.append(connector([(nodes_x, branch_top + 72), (nodes_x, branch_top + 104)]))
        body.append(connector([(nodes_x, branch_top + 176), (nodes_x, branch_top + 208)]))

    # Merge and output
    merge_y = 1198
    body.append(process_box(cx, merge_y, 420, 78, ["模型评价与集成复核", "Accuracy  BA  F1  混淆矩阵", "逐折交叉验证与一致性结果"], size=16, leading=22))
    body.append(process_box(cx, 1325, 420, 76, ["输出分类结果", "A：D1 D2 D3 D10", "B：D4 D5 D6  C：D7 D8 D9"], size=16, leading=22))
    body.append(process_box(cx, 1430, 300, 54, ["研究结论与生产监控建议"], size=17))

    left_bottom = branch_top + 280
    right_bottom = branch_top + 280
    body.append(connector([(left_x, left_bottom), (left_x, 1152), (cx - 60, 1152), (cx - 60, merge_y - 39)]))
    body.append(connector([(right_x, right_bottom), (right_x, 1152), (cx + 60, 1152), (cx + 60, merge_y - 39)]))
    body.append(connector([(cx, merge_y + 39), (cx, 1287)]))
    body.append(connector([(cx, 1363), (cx, 1403)]))

    save_svg(CHART_DIR / "08_modeling_flowchart.svg", width, height, "\n".join(body))


if __name__ == "__main__":
    main()
