from pathlib import Path
from html import escape as html_escape
import math

import numpy as np
import pandas as pd


ROOT = Path("/Users/kongfei/Documents/Codex/2026-06-05/a-b-c-x-specialdata-xlsx")
DATA_DIR = Path("/Users/kongfei/Desktop/2026年兰州大学数学建模竞赛赛题")
CHART_DIR = ROOT / "outputs" / "charts"
LABELS = ["A", "B", "C"]
COLORS = {
    "A": "#2A6FBB",
    "B": "#D98C1F",
    "C": "#2E9D64",
    "QDA": "#6E4AA8",
    "gray": "#5F6873",
    "dark": "#263238",
    "grid": "#E5E9F0",
}
FONT = "'Times New Roman', 'Songti SC', 'STSong', 'SimSun', serif"


def read_csv(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df.rename(columns={df.columns[0]: "id"})


def load_data():
    files = {
        "A": DATA_DIR / "SpecData_A类工件.csv",
        "B": DATA_DIR / "SpecData_B类工件.csv",
        "C": DATA_DIR / "SpecData_C类工件.csv",
    }
    frames = []
    for label, path in files.items():
        df = read_csv(path)
        df["label"] = label
        frames.append(df)
    train = pd.concat(frames, ignore_index=True)
    unknown = read_csv(DATA_DIR / "SpecData_未知类型工件.csv")
    feature_cols = list(unknown.columns[1:])
    return train, unknown, feature_cols


def class_arrays(train, feature_cols):
    return {
        label: train.loc[train["label"] == label, feature_cols].to_numpy(float)
        for label in LABELS
    }


def standardize(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X - mu) / sd


def pca_scores(X):
    Xz = standardize(X)
    _, s, vt = np.linalg.svd(Xz, full_matrices=False)
    eig = s ** 2 / (len(Xz) - 1)
    ratio = eig / eig.sum()
    scores = Xz @ vt.T
    return scores, ratio


def row_summary(X):
    return pd.DataFrame({
        "mean": X.mean(axis=1),
        "sd": X.std(axis=1, ddof=1),
        "range": X.max(axis=1) - X.min(axis=1),
        "iqr": np.percentile(X, 75, axis=1) - np.percentile(X, 25, axis=1),
    })


def esc(x):
    return html_escape(str(x), quote=True)


def fmt(x, digits=3):
    return f"{float(x):.{digits}f}"


def pct(x, digits=1):
    return f"{100 * float(x):.{digits}f}%"


def save_svg(path, width, height, body):
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <filter id="softShadow" x="-10%" y="-10%" width="120%" height="120%">
    <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="#000000" flood-opacity="0.12"/>
  </filter>
</defs>
<rect width="100%" height="100%" fill="#FFFFFF"/>
{body}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def text(x, y, content, size=14, weight=400, fill="#263238", anchor="start", extra=""):
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-family="{FONT}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}" {extra}>{esc(content)}</text>'
    )


def line(x1, y1, x2, y2, stroke="#263238", width=1, opacity=1, dash=None):
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
        f'stroke="{stroke}" stroke-width="{width}" opacity="{opacity}"{dash_attr}/>'
    )


def rect(x, y, w, h, fill="#FFFFFF", stroke="none", sw=1, rx=0, opacity=1, extra=""):
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" rx="{rx}" opacity="{opacity}" {extra}/>'
    )


def circle(cx, cy, r, fill, stroke="none", sw=1, opacity=1):
    return (
        f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}" opacity="{opacity}"/>'
    )


def polyline(points, stroke, width=2, opacity=1, fill="none"):
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return (
        f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{width}" '
        f'stroke-linecap="round" stroke-linejoin="round" opacity="{opacity}"/>'
    )


def polygon(points, fill, opacity=0.2, stroke="none", sw=0):
    pts = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    return f'<polygon points="{pts}" fill="{fill}" opacity="{opacity}" stroke="{stroke}" stroke-width="{sw}"/>'


def linear_scale(domain_min, domain_max, range_min, range_max):
    if abs(domain_max - domain_min) < 1e-12:
        return lambda _: (range_min + range_max) / 2
    return lambda v: range_min + (float(v) - domain_min) / (domain_max - domain_min) * (range_max - range_min)


def nice_ticks(lo, hi, count=5):
    if hi == lo:
        return [lo]
    raw = (hi - lo) / max(count - 1, 1)
    exp = math.floor(math.log10(abs(raw)))
    base = raw / (10 ** exp)
    if base <= 1:
        step = 1 * 10 ** exp
    elif base <= 2:
        step = 2 * 10 ** exp
    elif base <= 5:
        step = 5 * 10 ** exp
    else:
        step = 10 * 10 ** exp
    start = math.floor(lo / step) * step
    vals = []
    v = start
    while v <= hi + step * 0.5:
        if v >= lo - step * 0.5:
            vals.append(v)
        v += step
    return vals


def chart_title(width, title, subtitle=None):
    parts = [text(width / 2, 36, title, size=24, weight=700, anchor="middle")]
    if subtitle:
        parts.append(text(width / 2, 61, subtitle, size=13, fill="#5F6873", anchor="middle"))
    return "\n".join(parts)


def draw_axes(x0, y0, w, h, x_ticks, y_ticks, x_scale, y_scale, x_fmt=str, y_fmt=lambda v: fmt(v, 1)):
    parts = []
    parts.append(rect(x0, y0, w, h, fill="#FFFFFF", stroke="#D8DEE9", sw=1, rx=8))
    for yt in y_ticks:
        y = y_scale(yt)
        parts.append(line(x0, y, x0 + w, y, stroke=COLORS["grid"], width=1))
        parts.append(text(x0 - 10, y + 4, y_fmt(yt), size=11, fill="#60717F", anchor="end"))
    for xt in x_ticks:
        x = x_scale(xt)
        parts.append(line(x, y0 + h, x, y0 + h + 5, stroke="#60717F", width=1))
        parts.append(text(x, y0 + h + 22, x_fmt(xt), size=11, fill="#60717F", anchor="middle"))
    parts.append(line(x0, y0 + h, x0 + w, y0 + h, stroke="#263238", width=1.2))
    parts.append(line(x0, y0, x0, y0 + h, stroke="#263238", width=1.2))
    return "\n".join(parts)


def legend(x, y, items, gap=92):
    parts = []
    cx = x
    for label, color in items:
        parts.append(circle(cx, y - 5, 5, color))
        parts.append(text(cx + 10, y, label, size=12, fill="#37474F"))
        cx += gap
    return "\n".join(parts)


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return "#" + "".join(f"{int(max(0, min(255, c))):02x}" for c in rgb)


def lerp_color(c1, c2, t):
    a = hex_to_rgb(c1)
    b = hex_to_rgb(c2)
    return rgb_to_hex(tuple(a[i] + (b[i] - a[i]) * t for i in range(3)))
