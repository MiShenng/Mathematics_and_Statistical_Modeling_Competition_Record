from pathlib import Path
import json
import math

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "2026年兰州大学数学建模竞赛赛题" / "SpecData.xlsx"
OUT_DIR = BASE_DIR / "outputs"
LABELS = ["A", "B", "C"]
SHEETS = {"A": "A类工件", "B": "B类工件", "C": "C类工件", "unknown": "未知类型工件"}
SHRINK_GRID = [0.0, 0.02, 0.05, 0.08, 0.12, 0.20, 0.40, 0.60]


def read_sheet(sheet):
    df = pd.read_excel(DATA_FILE, sheet_name=sheet)
    return df.rename(columns={df.columns[0]: "id"})


def load_data():
    frames = []
    data_meta = {}
    for label in LABELS:
        df = read_sheet(SHEETS[label])
        features = list(df.columns[1:])
        data_meta[label] = {
            "samples": int(len(df)),
            "features": int(len(features)),
            "missing_values": int(df[features].isna().sum().sum()),
        }
        df = df.copy()
        df["label"] = label
        frames.append(df)
    train = pd.concat(frames, ignore_index=True)
    unknown = read_sheet(SHEETS["unknown"])
    feature_cols = list(unknown.columns[1:])
    return train, unknown, feature_cols, data_meta


def stratified_folds(y, k=5, seed=20260605):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for label in np.unique(y):
        idx = np.where(y == label)[0]
        rng.shuffle(idx)
        for fold_idx, part in enumerate(np.array_split(idx, k)):
            folds[fold_idx].extend(part.tolist())
    return [np.array(sorted(f), dtype=int) for f in folds]


def standardize_train_test(X_train, X_test):
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (X_train - mu) / sd, (X_test - mu) / sd


def class_stats(X, y):
    means, vars_, counts = {}, {}, {}
    for label in LABELS:
        Xi = X[y == label]
        means[label] = Xi.mean(axis=0)
        vars_[label] = Xi.var(axis=0, ddof=1)
        counts[label] = Xi.shape[0]
    return means, vars_, counts


def confusion_matrix(y_true, y_pred):
    mat = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    label_to_i = {label: i for i, label in enumerate(LABELS)}
    for yt, yp in zip(y_true, y_pred):
        mat[label_to_i[yt], label_to_i[yp]] += 1
    return mat


def metrics_from_confusion(cm):
    precision, recall, f1 = [], [], []
    for i in range(len(LABELS)):
        p = cm[i, i] / cm[:, i].sum() if cm[:, i].sum() else 0.0
        r = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        precision.append(float(p))
        recall.append(float(r))
        f1.append(float(f))
    return {
        "accuracy": float(np.trace(cm) / cm.sum()),
        "macro_precision": float(np.mean(precision)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "precision": dict(zip(LABELS, precision)),
        "recall": dict(zip(LABELS, recall)),
        "f1": dict(zip(LABELS, f1)),
        "confusion_matrix": cm.tolist(),
    }


def softmax(scores):
    z = scores - np.max(scores, axis=1, keepdims=True)
    e = np.exp(np.clip(z, -700, 700))
    return e / e.sum(axis=1, keepdims=True)


def predict_nearest_centroid(X_train, y_train, X_test):
    means, _, _ = class_stats(X_train, y_train)
    centroids = np.vstack([means[label] for label in LABELS])
    dist2 = ((X_test[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    pred = np.array([LABELS[i] for i in np.argmin(dist2, axis=1)])
    return pred, -dist2


def predict_nb(X_train, y_train, X_test, equal_prior=False):
    means, vars_, counts = class_stats(X_train, y_train)
    scores = []
    for label in LABELS:
        var = np.maximum(vars_[label], 1e-6)
        prior = 1 / len(LABELS) if equal_prior else counts[label] / len(y_train)
        score = math.log(prior)
        score += -0.5 * np.sum(np.log(var))
        score += -0.5 * ((X_test - means[label]) ** 2 / var).sum(axis=1)
        scores.append(score)
    scores = np.vstack(scores).T
    pred = np.array([LABELS[i] for i in np.argmax(scores, axis=1)])
    return pred, scores


def predict_lda(X_train, y_train, X_test, shrinkage=0.60, equal_prior=False):
    means, _, counts = class_stats(X_train, y_train)
    n, p = X_train.shape
    pooled = np.zeros((p, p))
    for label in LABELS:
        centered = X_train[y_train == label] - means[label]
        pooled += centered.T @ centered
    pooled /= max(n - len(LABELS), 1)
    cov = (1 - shrinkage) * pooled + shrinkage * np.diag(np.diag(pooled))
    cov += np.eye(p) * 1e-6
    inv_cov = np.linalg.pinv(cov)
    scores = []
    for label in LABELS:
        prior = 1 / len(LABELS) if equal_prior else counts[label] / len(y_train)
        mu = means[label]
        score = X_test @ inv_cov @ mu
        score += -0.5 * (mu @ inv_cov @ mu)
        score += math.log(prior)
        scores.append(score)
    scores = np.vstack(scores).T
    pred = np.array([LABELS[i] for i in np.argmax(scores, axis=1)])
    return pred, scores


def predict_qda(X_train, y_train, X_test, shrinkage=0.12, equal_prior=False):
    _, _, counts = class_stats(X_train, y_train)
    p = X_train.shape[1]
    scores = []
    for label in LABELS:
        Xi = X_train[y_train == label]
        mu = Xi.mean(axis=0)
        centered = Xi - mu
        cov = centered.T @ centered / max(Xi.shape[0] - 1, 1)
        cov = (1 - shrinkage) * cov + shrinkage * np.diag(np.diag(cov))
        cov += np.eye(p) * 1e-6
        inv_cov = np.linalg.pinv(cov)
        logdet = float(np.sum(np.log(np.maximum(np.linalg.eigvalsh(cov), 1e-12))))
        prior = 1 / len(LABELS) if equal_prior else counts[label] / len(y_train)
        diff = X_test - mu
        quad = np.einsum("ij,jk,ik->i", diff, inv_cov, diff)
        scores.append(-0.5 * quad - 0.5 * logdet + math.log(prior))
    scores = np.vstack(scores).T
    pred = np.array([LABELS[i] for i in np.argmax(scores, axis=1)])
    return pred, scores


def pca_project_train_test(X_train, X_test, r):
    center = X_train.mean(axis=0)
    Xc = X_train - center
    Xt = X_test - center
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    comps = vt[:r].T
    return Xc @ comps, Xt @ comps


def predict_pca_qda(X_train, y_train, X_test, r=12):
    Xtr_pc, Xte_pc = pca_project_train_test(X_train, X_test, r)
    return predict_qda(Xtr_pc, y_train, Xte_pc, shrinkage=0.10, equal_prior=False)


def cross_validate(X, y, model_name, predictor):
    folds = stratified_folds(y)
    y_true_all, y_pred_all, fold_acc = [], [], []
    for test_idx in folds:
        train_idx = np.setdiff1d(np.arange(len(y)), test_idx, assume_unique=True)
        Xtr, Xte = standardize_train_test(X[train_idx], X[test_idx])
        pred, _ = predictor(Xtr, y[train_idx], Xte)
        y_true_all.append(y[test_idx])
        y_pred_all.append(pred)
        fold_acc.append(float(np.mean(pred == y[test_idx])))
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    result = metrics_from_confusion(confusion_matrix(y_true, y_pred))
    result["model"] = model_name
    result["fold_accuracy"] = fold_acc
    return result


def row_summary(X):
    return pd.DataFrame({
        "mean": X.mean(axis=1),
        "sd": X.std(axis=1, ddof=1),
        "min": X.min(axis=1),
        "max": X.max(axis=1),
        "median": np.median(X, axis=1),
        "iqr": np.percentile(X, 75, axis=1) - np.percentile(X, 25, axis=1),
        "range": X.max(axis=1) - X.min(axis=1),
    })


def class_profile(train, feature_cols):
    rows = []
    for label in LABELS:
        Xg = train.loc[train["label"] == label, feature_cols].to_numpy(float)
        rs = row_summary(Xg)
        rows.append({
            "类别": label,
            "样本数": len(Xg),
            "全体均值": Xg.mean(),
            "全体标准差": Xg.std(ddof=1),
            "样本均值标准差": rs["mean"].std(ddof=1),
            "样本内标准差均值": rs["sd"].mean(),
            "最小观测值": Xg.min(),
            "最大观测值": Xg.max(),
        })
    return pd.DataFrame(rows)


def feature_anova(X, y, feature_cols):
    rows = []
    overall = X.mean(axis=0)
    total = ((X - overall) ** 2).sum(axis=0)
    for j, col in enumerate(feature_cols):
        groups = [X[y == label, j] for label in LABELS]
        means = [g.mean() for g in groups]
        between = sum(len(g) * (g.mean() - overall[j]) ** 2 for g in groups)
        within = sum(((g - g.mean()) ** 2).sum() for g in groups)
        F = (between / 2) / (within / (len(y) - 3))
        eta2 = between / total[j]
        rows.append({
            "检测位置": col,
            "A均值": means[0],
            "B均值": means[1],
            "C均值": means[2],
            "极差(类均值)": max(means) - min(means),
            "ANOVA_F": F,
            "eta2": eta2,
        })
    return pd.DataFrame(rows).sort_values("ANOVA_F", ascending=False)


def pca_summary(X, y):
    Xz = (X - X.mean(axis=0)) / np.where(X.std(axis=0, ddof=0) < 1e-12, 1, X.std(axis=0, ddof=0))
    _, s, vt = np.linalg.svd(Xz, full_matrices=False)
    eig = s ** 2 / (len(Xz) - 1)
    ratio = eig / eig.sum()
    scores = Xz @ vt[:3].T
    centers = {label: scores[y == label].mean(axis=0).tolist() for label in LABELS}
    corr = np.corrcoef(Xz, rowvar=False)
    adjacent_corr = float(np.mean([corr[i, i + 1] for i in range(corr.shape[0] - 1)]))
    offdiag_abs = float(np.abs(corr[np.triu_indices_from(corr, k=1)]).mean())
    return {
        "explained_ratio_first_10": ratio[:10].tolist(),
        "cumulative_first_10": np.cumsum(ratio)[:10].tolist(),
        "pc_centers": centers,
        "adjacent_corr": adjacent_corr,
        "offdiag_abs_corr": offdiag_abs,
    }


def best_ab_threshold(values, y):
    mask = (y == "A") | (y == "B")
    v = values[mask]
    yy = y[mask]
    order = np.argsort(v)
    vs = v[order]
    candidates = [(vs[i] + vs[i + 1]) / 2 for i in range(len(vs) - 1)]
    best = {"accuracy": -1}
    for t in candidates:
        pred = np.where(v > t, "A", "B")
        acc = float(np.mean(pred == yy))
        if acc > best["accuracy"]:
            best = {"threshold": float(t), "accuracy": acc}
    return best


def prior_sensitivity(X, y):
    rows = []
    models = {
        "NB": lambda equal: (lambda Xtr, ytr, Xte: predict_nb(Xtr, ytr, Xte, equal_prior=equal)),
        "LDA": lambda equal: (lambda Xtr, ytr, Xte: predict_lda(Xtr, ytr, Xte, equal_prior=equal)),
        "QDA": lambda equal: (lambda Xtr, ytr, Xte: predict_qda(Xtr, ytr, Xte, equal_prior=equal)),
    }
    for model, factory in models.items():
        for prior_name, equal in [("等先验", True), ("经验先验", False)]:
            r = cross_validate(X, y, model, factory(equal))
            rows.append({
                "模型": model,
                "先验": prior_name,
                "总准确率": r["accuracy"],
                "平衡准确率": r["balanced_accuracy"],
            })
    return pd.DataFrame(rows)


def shrinkage_sensitivity(X, y, predictor, name):
    """对收缩参数做网格敏感性分析，每个取值用同一分层五折交叉验证评价。"""
    rows = []
    for s in SHRINK_GRID:
        result = cross_validate(X, y, f"{name}({s})", lambda Xtr, ytr, Xte, s=s: predictor(Xtr, ytr, Xte, shrinkage=s))
        rows.append({"模型": name, "收缩参数": s, "总准确率": result["accuracy"], "平衡准确率": result["balanced_accuracy"]})
    return pd.DataFrame(rows)


def pca_r_sensitivity(X, y):
    rows = []
    for r in [1, 2, 3, 5, 8, 10, 12, 20, 30, 50]:
        result = cross_validate(X, y, f"PCA-QDA({r})", lambda Xtr, ytr, Xte, r=r: predict_pca_qda(Xtr, ytr, Xte, r=r))
        rows.append({"PCA维数": r, "总准确率": result["accuracy"], "平衡准确率": result["balanced_accuracy"]})
    return pd.DataFrame(rows)


def predict_unknown_all(X, y, unknown_X, unknown_ids):
    Xtr, Xu = standardize_train_test(X, unknown_X)
    predictors = {
        "最近质心": predict_nearest_centroid,
        "朴素贝叶斯": predict_nb,
        "LDA": lambda Xtr, ytr, Xte: predict_lda(Xtr, ytr, Xte, shrinkage=0.60, equal_prior=False),
        "QDA": lambda Xtr, ytr, Xte: predict_qda(Xtr, ytr, Xte, shrinkage=0.12, equal_prior=False),
        "PCA-QDA": lambda Xtr, ytr, Xte: predict_pca_qda(Xtr, ytr, Xte, r=12),
    }
    outputs = {}
    qda_prob = None
    for name, fn in predictors.items():
        pred, scores = fn(Xtr, y, Xu)
        outputs[name] = pred
        if name == "QDA":
            qda_prob = softmax(scores).max(axis=1)
    rows = []
    for i, item_id in enumerate(unknown_ids):
        votes = [outputs[name][i] for name in predictors]
        counts = {lab: votes.count(lab) for lab in LABELS}
        top = max(counts.values())
        winners = [lab for lab in LABELS if counts[lab] == top]
        # 平票时以主分类器 QDA 的判别为准
        final = outputs["QDA"][i] if len(winners) > 1 else winners[0]
        rows.append({
            "工件序号": item_id,
            "预测类别": final,
            "一致模型数": sum(v == final for v in votes),
            "最近质心": outputs["最近质心"][i],
            "朴素贝叶斯": outputs["朴素贝叶斯"][i],
            "LDA": outputs["LDA"][i],
            "QDA": outputs["QDA"][i],
            "PCA-QDA": outputs["PCA-QDA"][i],
            "QDA最大后验概率": qda_prob[i],
        })
    return pd.DataFrame(rows)


def md_table(df, float_digits=4):
    headers = list(df.columns)
    rows = []
    for _, row in df.iterrows():
        vals = []
        for col in headers:
            value = row[col]
            if col in {"PCA维数", "一致模型数", "样本数"}:
                vals.append(str(int(round(float(value)))))
            elif isinstance(value, (float, np.floating)):
                vals.append(f"{float(value):.{float_digits}f}")
            else:
                vals.append(str(value))
        rows.append(vals)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for vals in rows:
        out.append("| " + " | ".join(vals) + " |")
    return "\n".join(out)


def write_markdown_report(paths, profile, feature_top, pca, cv_df, class_metric_df, fold_df,
                          prior_df, lda_grid_df, qda_grid_df, pca_r_df, unknown_df, ab_threshold, std_stats):
    lines = []
    lines.append("# 工件分类数学建模最终作答\n")
    lines.append("## 步骤 1：数据读取与质量检查\n")
    lines.append("训练集规模为 A 类 3572、B 类 2643、C 类 1763，共 7978 个样本；未知样本 10 个。每个工件有 155 个检测位置。三类训练数据缺失值均为 0。\n")
    lines.append(md_table(profile, float_digits=4))
    lines.append("\n\n## 步骤 2：特征分析\n")
    lines.append(md_table(feature_top.head(12), float_digits=4))
    lines.append("\n\nPCA 前 3 个主成分解释率分别为 "
                 f"{pca['explained_ratio_first_10'][0]:.4f}、{pca['explained_ratio_first_10'][1]:.4f}、{pca['explained_ratio_first_10'][2]:.4f}，"
                 f"前 10 个累计解释率为 {pca['cumulative_first_10'][9]:.4f}。\n")
    lines.append("\n## 步骤 3：模型构建与交叉验证\n")
    lines.append(md_table(cv_df, float_digits=6))
    lines.append("\n\n分类别 Precision/F1：\n")
    lines.append(md_table(class_metric_df, float_digits=6))
    lines.append("\n\n各折准确率：\n")
    lines.append(md_table(fold_df, float_digits=6))
    lines.append("\n\n## 步骤 4：敏感性分析\n")
    lines.append("先验概率敏感性：\n")
    lines.append(md_table(prior_df, float_digits=6))
    lines.append("\n\nLDA 收缩参数敏感性：\n")
    lines.append(md_table(lda_grid_df, float_digits=6))
    lines.append("\n\nQDA 收缩参数敏感性：\n")
    lines.append(md_table(qda_grid_df, float_digits=6))
    lines.append("\n\nPCA 维数敏感性：\n")
    lines.append(md_table(pca_r_df, float_digits=6))
    lines.append(f"\n\nA/B 样本内标准差最优切割点为 {ab_threshold['threshold']:.6f}，"
                 f"A 类样本内标准差最小值为 {std_stats['A_min']:.6f}，B 类最大值为 {std_stats['B_max']:.6f}。\n")
    lines.append("\n## 步骤 5：未知工件分类\n")
    lines.append(md_table(unknown_df, float_digits=6))
    lines.append("\n\n最终分类：D1、D2、D3、D10 为 A；D4、D5、D6 为 B；D7、D8、D9 为 C。\n")
    lines.append("\n## 输出文件\n")
    for name, path in paths.items():
        lines.append(f"- {name}: `{path}`")
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train, unknown, feature_cols, data_meta = load_data()
    X = train[feature_cols].to_numpy(float)
    y = train["label"].to_numpy()
    unknown_X = unknown[feature_cols].to_numpy(float)
    unknown_ids = unknown["id"].astype(str).tolist()

    profile = class_profile(train, feature_cols)
    feature_rank = feature_anova(X, y, feature_cols)
    pca = pca_summary(X, y)

    model_specs = [
        ("最近质心", predict_nearest_centroid),
        ("高斯朴素贝叶斯", lambda Xtr, ytr, Xte: predict_nb(Xtr, ytr, Xte, equal_prior=False)),
        ("正则化LDA", lambda Xtr, ytr, Xte: predict_lda(Xtr, ytr, Xte, shrinkage=0.60, equal_prior=False)),
        ("正则化QDA", lambda Xtr, ytr, Xte: predict_qda(Xtr, ytr, Xte, shrinkage=0.12, equal_prior=False)),
        ("PCA-QDA(12维)", lambda Xtr, ytr, Xte: predict_pca_qda(Xtr, ytr, Xte, r=12)),
    ]
    cv_results = [cross_validate(X, y, name, fn) for name, fn in model_specs]
    cv_df = pd.DataFrame([{
        "模型": r["model"], "总准确率": r["accuracy"], "宏精确率": r["macro_precision"],
        "平衡准确率": r["balanced_accuracy"], "宏F1": r["macro_f1"],
        "A召回率": r["recall"]["A"], "B召回率": r["recall"]["B"], "C召回率": r["recall"]["C"],
    } for r in cv_results])
    class_metric_df = pd.DataFrame([{
        "模型": r["model"], "A精确率": r["precision"]["A"], "A_F1": r["f1"]["A"],
        "B精确率": r["precision"]["B"], "B_F1": r["f1"]["B"],
        "C精确率": r["precision"]["C"], "C_F1": r["f1"]["C"],
    } for r in cv_results])
    fold_df = pd.DataFrame([{
        "模型": r["model"], "Fold1": r["fold_accuracy"][0], "Fold2": r["fold_accuracy"][1],
        "Fold3": r["fold_accuracy"][2], "Fold4": r["fold_accuracy"][3], "Fold5": r["fold_accuracy"][4],
    } for r in cv_results])
    prior_df = prior_sensitivity(X, y)
    lda_grid_df = shrinkage_sensitivity(X, y, predict_lda, "LDA")
    qda_grid_df = shrinkage_sensitivity(X, y, predict_qda, "QDA")
    pca_r_df = pca_r_sensitivity(X, y)
    unknown_pred = predict_unknown_all(X, y, unknown_X, unknown_ids)

    rs = row_summary(X)
    ab_threshold = best_ab_threshold(rs["sd"].to_numpy(), y)
    std_stats = {
        "A_min": float(rs.loc[y == "A", "sd"].min()),
        "B_max": float(rs.loc[y == "B", "sd"].max()),
    }

    paths = {
        "最终作答": OUT_DIR / "final_modeling_answer.md",
        "未知样本分类": OUT_DIR / "final_unknown_classification.csv",
        "模型验证指标": OUT_DIR / "final_model_metrics.csv",
        "关键检测位置": OUT_DIR / "final_feature_importance.csv",
        "收缩参数敏感性": OUT_DIR / "final_shrinkage_grid.csv",
        "机器可读结果": OUT_DIR / "final_results.json",
    }

    unknown_pred.to_csv(paths["未知样本分类"], index=False, encoding="utf-8-sig")
    cv_df.to_csv(paths["模型验证指标"], index=False, encoding="utf-8-sig")
    feature_rank.head(30).to_csv(paths["关键检测位置"], index=False, encoding="utf-8-sig")
    pd.concat([lda_grid_df, qda_grid_df], ignore_index=True).to_csv(paths["收缩参数敏感性"], index=False, encoding="utf-8-sig")

    result_json = {
        "data_meta": data_meta,
        "pca": pca,
        "cv_results": cv_results,
        "lda_shrinkage_grid": lda_grid_df.to_dict(orient="records"),
        "qda_shrinkage_grid": qda_grid_df.to_dict(orient="records"),
        "ab_std_threshold": ab_threshold,
        "std_stats": std_stats,
        "unknown_predictions": unknown_pred.to_dict(orient="records"),
    }
    paths["机器可读结果"].write_text(json.dumps(result_json, ensure_ascii=False, indent=2), encoding="utf-8")

    report = write_markdown_report(
        {k: str(v) for k, v in paths.items()},
        profile, feature_rank, pca, cv_df, class_metric_df, fold_df,
        prior_df, lda_grid_df, qda_grid_df, pca_r_df, unknown_pred, ab_threshold, std_stats,
    )
    paths["最终作答"].write_text(report, encoding="utf-8")

    print("完成。主要输出：")
    for key, path in paths.items():
        print(f"{key}: {path}")
    print("\n未知样本分类：")
    print(unknown_pred[["工件序号", "预测类别", "一致模型数"]].to_string(index=False))


if __name__ == "__main__":
    main()
