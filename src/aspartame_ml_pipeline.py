from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.svm import SVC


SHEETS = {
    "Pristine Graphene": "Pristine Graphene",
    "AptamerAuGraphene": "Aptamer/Au/Graphene",
}

ASPARTAME_FREE = {"Coke", "Pepsi", "Redbull", "Snapple", "MinuteMaid"}
ASPARTAME_CONTAINING = {
    "Coke Zero",
    "Pepsi Zero",
    "Redbull Sugarfree",
    "Snapple Zero",
    "MinuteMaid Zero",
}

DRINK_FAMILIES = {
    "Coke": "Coke",
    "Coke Zero": "Coke",
    "Pepsi": "Pepsi",
    "Pepsi Zero": "Pepsi",
    "Redbull": "Redbull",
    "Redbull Sugarfree": "Redbull",
    "Snapple": "Snapple",
    "Snapple Zero": "Snapple",
    "MinuteMaid": "MinuteMaid",
    "MinuteMaid Zero": "MinuteMaid",
}

LABELS = ["Aspartame-free", "Aspartame-containing"]
SENSORS = ["Pristine Graphene", "Aptamer/Au/Graphene"]
META_COLS = {"sample_id", "sensor", "sheet", "drink", "drink_family", "label", "label_name", "pulse"}


def safe_auc(t: np.ndarray, y: np.ndarray) -> float:
    if len(t) < 2:
        return float("nan")
    return float(np.trapezoid(y, t))


def sample_ids(df: pd.DataFrame) -> pd.Series:
    return (
        df["sensor"].str.replace("/", "-", regex=False).str.replace(" ", "_", regex=False)
        + "__"
        + df["drink"].str.replace(" ", "_", regex=False)
        + "__pulse_"
        + df["pulse"].astype(str).str.zfill(2)
    )


def extract_window_features(
    time: np.ndarray,
    current: np.ndarray,
    window_start: float,
    window_end: float,
) -> dict[str, float] | None:
    mask = (time >= window_start) & (time < window_end)
    t = time[mask]
    i = current[mask]
    good = np.isfinite(t) & np.isfinite(i)
    t = t[good]
    i = i[good]
    if len(t) < 20:
        return None

    order = np.argsort(t)
    t = t[order]
    i = i[order]
    local_t = t - window_start
    duration = max(window_end - window_start, 1.0)

    baseline_cut = window_start + 0.10 * duration
    baseline_vals = i[t <= baseline_cut]
    if len(baseline_vals) < 5:
        baseline_vals = i[: max(5, int(0.10 * len(i)))]
    baseline = float(np.median(baseline_vals))
    if not np.isfinite(baseline) or abs(baseline) < 1e-15:
        return None

    response = (i - baseline) / abs(baseline) * 100.0
    response = response - float(np.median(response[: max(5, int(0.05 * len(response)))]))

    max_idx = int(np.argmax(response))
    min_idx = int(np.argmin(response))
    peak = float(response[max_idx])
    trough = float(response[min_idx])
    abs_peak = peak if abs(peak) >= abs(trough) else trough
    abs_peak_idx = max_idx if abs(peak) >= abs(trough) else min_idx

    positive = np.clip(response, 0, None)
    negative = np.clip(-response, 0, None)
    end_slice = response[int(0.90 * len(response)) :] if len(response) > 10 else response[-5:]
    pre_slice = response[: max(5, int(0.10 * len(response)))]
    diffs = np.diff(response) / np.maximum(np.diff(local_t), 1e-9)

    return {
        "baseline_current": baseline,
        "peak_response": peak,
        "trough_response": trough,
        "abs_peak_response": float(abs(abs_peak)),
        "peak_time_s": float(local_t[abs_peak_idx]),
        "mean_response": float(np.mean(response)),
        "median_response": float(np.median(response)),
        "std_response": float(np.std(response, ddof=1)),
        "auc_response": safe_auc(local_t, response),
        "positive_auc": safe_auc(local_t, positive),
        "negative_auc": safe_auc(local_t, negative),
        "range_response": float(np.max(response) - np.min(response)),
        "end_response": float(np.median(end_slice)),
        "recovery_ratio": float(np.median(end_slice) / peak) if abs(peak) > 1e-9 else 0.0,
        "initial_noise": float(np.std(pre_slice, ddof=1)) if len(pre_slice) > 1 else 0.0,
        "snr": float(abs(peak) / (np.std(pre_slice, ddof=1) + 1e-9)) if len(pre_slice) > 1 else 0.0,
        "max_slope": float(np.max(diffs)) if len(diffs) else 0.0,
        "min_slope": float(np.min(diffs)) if len(diffs) else 0.0,
    }


def load_features(
    input_xlsx: Path,
    n_pulses: int,
    time_start: float,
    time_end: float,
) -> pd.DataFrame:
    rows = []
    xl = pd.ExcelFile(input_xlsx)
    edges = np.linspace(time_start, time_end, n_pulses + 1)

    for sheet, sensor in SHEETS.items():
        df = pd.read_excel(xl, sheet_name=sheet, header=None)
        for c in range(df.shape[1] - 1):
            if str(df.iat[3, c]).strip() != "Time (s)":
                continue

            drink = str(df.iat[2, c]).strip()
            if drink in ASPARTAME_FREE:
                label = 0
                label_name = LABELS[0]
            elif drink in ASPARTAME_CONTAINING:
                label = 1
                label_name = LABELS[1]
            else:
                continue

            time = pd.to_numeric(df.iloc[4:, c], errors="coerce").to_numpy(dtype=float)
            current = pd.to_numeric(df.iloc[4:, c + 1], errors="coerce").to_numpy(dtype=float)
            for pulse_idx in range(n_pulses):
                feats = extract_window_features(time, current, edges[pulse_idx], edges[pulse_idx + 1])
                if feats is None:
                    continue
                feats.update(
                    {
                        "sensor": sensor,
                        "sheet": sheet,
                        "drink": drink,
                        "drink_family": DRINK_FAMILIES[drink],
                        "label": label,
                        "label_name": label_name,
                        "pulse": pulse_idx + 1,
                    }
                )
                rows.append(feats)

    features = pd.DataFrame(rows)
    features.insert(0, "sample_id", sample_ids(features))
    return features


def feature_columns(features: pd.DataFrame) -> list[str]:
    return [c for c in features.columns if c not in META_COLS]


def classifier_specs() -> dict[str, Pipeline]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=7)),
            ]
        ),
        "Linear SVM": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVC(kernel="linear", C=1.0, class_weight="balanced", probability=True, random_state=7)),
            ]
        ),
        "RBF SVM": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", SVC(kernel="rbf", C=2.0, gamma="scale", class_weight="balanced", probability=True, random_state=7)),
            ]
        ),
        "Random Forest": Pipeline(
            [
                ("scale", RobustScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=4,
                        min_samples_leaf=3,
                        class_weight="balanced",
                        random_state=7,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", GradientBoostingClassifier(random_state=7, n_estimators=120, learning_rate=0.04, max_depth=2)),
            ]
        ),
        "LDA": Pipeline([("scale", StandardScaler()), ("model", LinearDiscriminantAnalysis())]),
    }


def scorer_auc(estimator: Pipeline, x: np.ndarray, y: np.ndarray) -> float:
    return roc_auc_score(y, estimator.predict_proba(x)[:, 1])


def evaluate_random_cv(features: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=7)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
        "roc_auc": scorer_auc,
    }

    for sensor, sdf in features.groupby("sensor", sort=False):
        x = sdf[cols].to_numpy(dtype=float)
        y = sdf["label"].to_numpy(dtype=int)
        for model_name, model in classifier_specs().items():
            scores = cross_validate(model, x, y, cv=cv, scoring=scoring, n_jobs=None)
            row = {
                "validation_setting": "random_repeated_5fold",
                "metric_level": "model_summary",
                "model": model_name,
                "sensor": sensor,
                "held_out_family": "ALL",
                "n": len(sdf),
                "features": len(cols),
            }
            for metric in scoring:
                vals = scores[f"test_{metric}"]
                row[metric] = float(np.mean(vals))
                row[f"{metric}_std"] = float(np.std(vals, ddof=1))
            rows.append(row)
    return pd.DataFrame(rows)


def evaluate_leave_family_out(features: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    fold_rows = []
    families = ["Coke", "Pepsi", "Redbull", "Snapple", "MinuteMaid"]

    for sensor, sdf in features.groupby("sensor", sort=False):
        x_all = sdf[cols].to_numpy(dtype=float)
        y_all = sdf["label"].to_numpy(dtype=int)
        for model_name in classifier_specs().keys():
            y_true_all = []
            y_pred_all = []
            y_prob_all = []
            for family in families:
                test_mask = (sdf["drink_family"] == family).to_numpy()
                train_mask = ~test_mask
                model = classifier_specs()[model_name]
                model.fit(x_all[train_mask], y_all[train_mask])
                y_test = y_all[test_mask]
                y_pred = model.predict(x_all[test_mask])
                y_prob = model.predict_proba(x_all[test_mask])[:, 1]
                fold_rows.append(
                    metric_row(
                        "leave_family_out",
                        "fold",
                        model_name,
                        sensor,
                        family,
                        y_test,
                        y_pred,
                        y_prob,
                    )
                )
                y_true_all.extend(y_test.tolist())
                y_pred_all.extend(y_pred.tolist())
                y_prob_all.extend(y_prob.tolist())
            rows.append(
                metric_row(
                    "leave_family_out",
                    "model_summary",
                    model_name,
                    sensor,
                    "ALL",
                    np.array(y_true_all),
                    np.array(y_pred_all),
                    np.array(y_prob_all),
                )
            )
    return pd.DataFrame(rows), pd.DataFrame(fold_rows)


def metric_row(
    validation_setting: str,
    metric_level: str,
    model: str,
    sensor: str,
    held_out_family: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float | str | int]:
    return {
        "validation_setting": validation_setting,
        "metric_level": metric_level,
        "model": model,
        "sensor": sensor,
        "held_out_family": held_out_family,
        "n": int(len(y_true)),
        "features": "",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_std": "",
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "balanced_accuracy_std": "",
        "f1": float(f1_score(y_true, y_pred)),
        "f1_std": "",
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "roc_auc_std": "",
    }


def selected_model_predictions(features: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = {"Gradient Boosting": classifier_specs()["Gradient Boosting"], "LDA": classifier_specs()["LDA"]}
    pred_rows = []
    conf_rows = []
    families = ["Coke", "Pepsi", "Redbull", "Snapple", "MinuteMaid"]

    for sensor, sdf in features.groupby("sensor", sort=False):
        sdf = sdf.reset_index(drop=True)
        x = sdf[cols].to_numpy(dtype=float)
        y = sdf["label"].to_numpy(dtype=int)

        for model_name, model_template in selected.items():
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=11)
            random_pred = np.zeros_like(y)
            random_prob = np.zeros(len(y), dtype=float)
            for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
                model = classifier_specs()[model_name]
                model.fit(x[train_idx], y[train_idx])
                y_pred = model.predict(x[test_idx])
                y_prob = model.predict_proba(x[test_idx])[:, 1]
                random_pred[test_idx] = y_pred
                random_prob[test_idx] = y_prob
                for local_i, idx in enumerate(test_idx):
                    pred_rows.append(prediction_row("random_5fold", model_name, fold, "ALL", sdf, idx, y_pred[local_i], y_prob[local_i]))
            conf_rows.extend(confusion_rows("random_5fold", model_name, "overall", "ALL", sensor, y, random_pred))

            leave_pred = np.zeros_like(y)
            leave_prob = np.zeros(len(y), dtype=float)
            for family in families:
                test_mask = (sdf["drink_family"] == family).to_numpy()
                train_mask = ~test_mask
                model = classifier_specs()[model_name]
                model.fit(x[train_mask], y[train_mask])
                y_pred = model.predict(x[test_mask])
                y_prob = model.predict_proba(x[test_mask])[:, 1]
                test_indices = np.where(test_mask)[0]
                leave_pred[test_indices] = y_pred
                leave_prob[test_indices] = y_prob
                for local_i, idx in enumerate(test_indices):
                    pred_rows.append(prediction_row("leave_family_out", model_name, "", family, sdf, idx, y_pred[local_i], y_prob[local_i]))
                conf_rows.extend(confusion_rows("leave_family_out", model_name, "by_family", family, sensor, y[test_mask], y_pred))
            conf_rows.extend(confusion_rows("leave_family_out", model_name, "overall", "ALL", sensor, y, leave_pred))

    return pd.DataFrame(pred_rows), pd.DataFrame(conf_rows)


def prediction_row(
    validation_setting: str,
    model: str,
    fold: int | str,
    held_out_family: str,
    df: pd.DataFrame,
    idx: int,
    predicted_label: int,
    prob: float,
) -> dict[str, str | int | float]:
    return {
        "validation_setting": validation_setting,
        "model": model,
        "fold": fold,
        "held_out_family": held_out_family,
        "sample_id": df.loc[idx, "sample_id"],
        "sensor": df.loc[idx, "sensor"],
        "drink": df.loc[idx, "drink"],
        "drink_family": df.loc[idx, "drink_family"],
        "pulse": int(df.loc[idx, "pulse"]),
        "true_label": LABELS[int(df.loc[idx, "label"])],
        "predicted_label": LABELS[int(predicted_label)],
        "prob_aspartame_containing": float(prob),
    }


def confusion_rows(
    validation_setting: str,
    model: str,
    aggregation: str,
    held_out_family: str,
    sensor: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, str | int | float]]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    total = tn + fp + fn + tp
    return [
        {
            "validation_setting": validation_setting,
            "model": model,
            "aggregation": aggregation,
            "held_out_family": held_out_family,
            "sensor": sensor,
            "TN_free_correct": tn,
            "FP_free_as_containing": fp,
            "FN_containing_as_free": fn,
            "TP_containing_correct": tp,
            "total": total,
            "accuracy": (tn + tp) / total if total else np.nan,
            "sensitivity_containing": tp / (tp + fn) if (tp + fn) else np.nan,
            "specificity_free": tn / (tn + fp) if (tn + fp) else np.nan,
        }
    ]


def build_embeddings(features: pd.DataFrame, cols: list[str], output_dir: Path) -> pd.DataFrame:
    rows = []
    for sensor in SENSORS:
        sdf = features[features["sensor"] == sensor].copy().reset_index(drop=True)
        x_scaled = StandardScaler().fit_transform(sdf[cols].to_numpy(dtype=float))
        pca = PCA(n_components=2, random_state=7)
        pca_xy = pca.fit_transform(x_scaled)
        tsne_xy = TSNE(n_components=2, perplexity=25, init="pca", learning_rate="auto", random_state=7).fit_transform(x_scaled)

        for idx, row in sdf.iterrows():
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "sensor": sensor,
                    "drink": row["drink"],
                    "drink_family": row["drink_family"],
                    "label": int(row["label"]),
                    "label_name": row["label_name"],
                    "pulse": int(row["pulse"]),
                    "pca1": float(pca_xy[idx, 0]),
                    "pca2": float(pca_xy[idx, 1]),
                    "tsne1": float(tsne_xy[idx, 0]),
                    "tsne2": float(tsne_xy[idx, 1]),
                    "pca_explained_variance_ratio_1": float(pca.explained_variance_ratio_[0]),
                    "pca_explained_variance_ratio_2": float(pca.explained_variance_ratio_[1]),
                }
            )

    embed = pd.DataFrame(rows)
    embed.to_csv(output_dir / "embedding_pca_tsne_coordinates.csv", index=False, encoding="utf-8-sig")
    centroid = (
        embed.groupby(["sensor", "label_name", "drink"], sort=False)[["pca1", "pca2", "tsne1", "tsne2"]]
        .mean()
        .reset_index()
    )
    centroid.to_csv(output_dir / "embedding_pca_tsne_centroids.csv", index=False, encoding="utf-8-sig")
    return embed


def plot_embeddings(embed: pd.DataFrame, output_dir: Path) -> None:
    colors = {LABELS[0]: "#3d6fb6", LABELS[1]: "#d95f5f"}
    for method, xcol, ycol, filename in [
        ("PCA", "pca1", "pca2", "fig_pca_by_sensor.png"),
        ("t-SNE", "tsne1", "tsne2", "fig_tsne_by_sensor.png"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        for ax, sensor in zip(axes, SENSORS):
            sdf = embed[embed["sensor"] == sensor]
            for label in LABELS:
                ldf = sdf[sdf["label_name"] == label]
                ax.scatter(ldf[xcol], ldf[ycol], s=34, alpha=0.72, c=colors[label], label=label, edgecolor="white", linewidth=0.35)
            ax.set_title(sensor)
            ax.set_xlabel(f"{method} 1")
            ax.set_ylabel(f"{method} 2")
            ax.grid(alpha=0.2)
        axes[1].legend(frameon=False, loc="best")
        fig.suptitle(f"{method} feature-space distribution")
        fig.savefig(output_dir / filename, dpi=220)
        plt.close(fig)


def plot_confusion(conf: pd.DataFrame, output_dir: Path, validation_setting: str, model: str, filename: str) -> None:
    sdf = conf[(conf["validation_setting"] == validation_setting) & (conf["model"] == model) & (conf["aggregation"] == "overall")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, sensor in zip(axes, SENSORS):
        row = sdf[sdf["sensor"] == sensor].iloc[0]
        matrix = np.array(
            [
                [row["TN_free_correct"], row["FP_free_as_containing"]],
                [row["FN_containing_as_free"], row["TP_containing_correct"]],
            ],
            dtype=int,
        )
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title(sensor)
        ax.set_xticks([0, 1], LABELS, rotation=25, ha="right")
        ax.set_yticks([0, 1], LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=axes, shrink=0.75)
    fig.suptitle(f"{validation_setting}: {model}")
    fig.savefig(output_dir / filename, dpi=220)
    plt.close(fig)


def write_manifest(output_dir: Path) -> None:
    manifest = {
        "package": "Aspartame beverage sensor ML outputs",
        "description": "Merged package without SHAP: features, embeddings, metrics, predictions, confusion matrices, sample counts, figures.",
        "normalization": [
            "Each pulse was baseline-normalized as Response (%) = (I - I0) / |I0| * 100.",
            "Feature scaling was fitted inside each training split for model evaluation.",
        ],
        "files": sorted(p.name for p in output_dir.iterdir() if p.is_file() and p.suffix != ".zip"),
    }
    with open(output_dir / "MANIFEST.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def zip_outputs(output_dir: Path) -> Path:
    zip_path = output_dir / "result_a_merged_data_package.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output_dir.iterdir()):
            if path.is_file() and path.name != zip_path.name:
                zf.write(path, arcname=path.name)
    return zip_path


def run_pipeline(input_xlsx: Path, output_dir: Path, n_pulses: int, time_start: float, time_end: float) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = load_features(input_xlsx, n_pulses=n_pulses, time_start=time_start, time_end=time_end)
    cols = feature_columns(features)
    features.to_csv(output_dir / "features_all_pulses.csv", index=False, encoding="utf-8-sig")

    counts = features.groupby(["sensor", "label_name", "drink"], sort=False).size().reset_index(name="pulse_count")
    counts.to_csv(output_dir / "sample_counts.csv", index=False, encoding="utf-8-sig")

    random_metrics = evaluate_random_cv(features, cols)
    leave_metrics, leave_fold_metrics = evaluate_leave_family_out(features, cols)
    metrics = pd.concat([random_metrics, leave_metrics, leave_fold_metrics], ignore_index=True, sort=False)
    metrics.to_csv(output_dir / "metrics_all.csv", index=False, encoding="utf-8-sig")

    predictions, confusions = selected_model_predictions(features, cols)
    predictions.to_csv(output_dir / "predictions_all_models.csv", index=False, encoding="utf-8-sig")
    confusions.to_csv(output_dir / "confusion_matrices_all.csv", index=False, encoding="utf-8-sig")

    embed = build_embeddings(features, cols, output_dir)
    plot_embeddings(embed, output_dir)
    plot_confusion(confusions, output_dir, "random_5fold", "Gradient Boosting", "fig_confusion_random_5fold_gradient_boosting.png")
    plot_confusion(confusions, output_dir, "leave_family_out", "Gradient Boosting", "fig_confusion_leave_family_out_gradient_boosting.png")

    write_manifest(output_dir)
    zip_path = zip_outputs(output_dir)
    print(f"Saved outputs to: {output_dir}")
    print(f"Saved ZIP package: {zip_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aspartame beverage sensor ML pipeline")
    parser.add_argument("--input-xlsx", required=True, type=Path, help="Path to the raw Excel workbook.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for generated outputs.")
    parser.add_argument("--n-pulses", default=20, type=int, help="Number of pulses per beverage.")
    parser.add_argument("--time-start", default=0.0, type=float, help="Start time for pulse segmentation.")
    parser.add_argument("--time-end", default=6000.0, type=float, help="End time for pulse segmentation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(args.input_xlsx, args.output_dir, args.n_pulses, args.time_start, args.time_end)


if __name__ == "__main__":
    main()
