# Aspartame Beverage Sensor Machine Learning

This repository contains a Python pipeline for pulse-level machine learning analysis of beverage sensor data.

The analysis compares two sensor types:

- Pristine Graphene
- Aptamer/Au/Graphene

The classification task is binary:

- `0`: aspartame-free beverage
- `1`: aspartame-containing beverage

## Workflow

The pipeline performs:

1. Load raw Excel time-current data.
2. Segment each beverage response into 20 pulse-level samples.
3. Normalize each pulse using its own baseline current.
4. Extract time-domain pulse features.
5. Train and evaluate machine learning classifiers.
6. Run two validation strategies:
   - random repeated stratified 5-fold cross-validation
   - leave-beverage-family-out validation
7. Generate PCA and t-SNE coordinates and figures.
8. Export merged CSV source data, confusion matrices, metrics, predictions, and a ZIP package.

SHAP analysis is intentionally not included.

## Input Excel Format

The default sheet names expected by the script are:

- `Pristine Graphene`
- `AptamerAuGraphene`

Each beverage should be arranged as paired columns:

- `Time (s)`
- `Drain Current (A)`

The script expects the beverage names in row 3 and the column labels in row 4, matching the provided experimental workbook format.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/aspartame_ml_pipeline.py --input-xlsx path/to/aspartame_beverage_data.xlsx --output-dir outputs/result_a
```

Optional arguments:

```bash
python src/aspartame_ml_pipeline.py \
  --input-xlsx path/to/aspartame_beverage_data.xlsx \
  --output-dir outputs/result_a \
  --n-pulses 20 \
  --time-start 0 \
  --time-end 6000
```

## Main Outputs

- `features_all_pulses.csv`: pulse-level feature table
- `sample_counts.csv`: sample count by sensor, label, and beverage
- `metrics_all.csv`: model metrics for random CV and leave-family-out validation
- `predictions_all_models.csv`: sample-level predictions for selected models
- `confusion_matrices_all.csv`: simplified confusion matrix table
- `embedding_pca_tsne_coordinates.csv`: PCA/t-SNE coordinates for each pulse
- `embedding_pca_tsne_centroids.csv`: drink-level embedding centroids
- `fig_pca_by_sensor.png`: PCA scatter plot
- `fig_tsne_by_sensor.png`: t-SNE scatter plot
- `fig_confusion_random_5fold_gradient_boosting.png`: confusion matrix figure
- `fig_confusion_leave_family_out_gradient_boosting.png`: confusion matrix figure
- `result_a_merged_data_package.zip`: compressed output package

