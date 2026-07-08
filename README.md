# Prudential Life Insurance Assessment

This project builds a binary classification pipeline that predicts whether an applicant for life insurance falls into a higher or lower risk group, based on the Prudential Life Insurance Assessment dataset. The original target variable (`Response`) has 8 ordinal risk classes, and the notebook collapses these into a binary label (0 for classes 1 to 4, 1 for classes 5 to 8) to simplify the modeling problem into a risk classification task.

The notebook solves the problem of turning a large, messy insurance application dataset (with missing values, mixed categorical and numeric fields, and high-cardinality categories) into a clean, model-ready feature set, and then compares three different classifiers to find which one best separates high risk from low risk applicants.

It is intended for data scientists or analysts who want a worked example of an end-to-end tabular ML pipeline: data cleaning, feature engineering, encoding strategies for categorical variables of different cardinality, feature selection based on importance, and model comparison with cross-validated hyperparameter tuning.

## Features

- Loads and inspects the first 59,000 rows of `train.csv`, including shape, summary statistics, and missing value counts.
- Converts the original 8-class `Response` target into a binary target (`Response_bin`).
- Drops the rarest categorical columns per the project's data preparation instructions.
- Engineers new features:
  - `med_keyword_sum`: row-wise sum across all `Medical_Keyword_*` columns.
  - `med_history_sum`: row-wise sum across all `Medical_History_*` columns (NaNs treated as 0).
  - `family_hist_sum`: row-wise sum across all `Family_Hist_*` columns.
  - `bmi_age`: interaction term between `BMI` and `Ins_Age`.
  - `height_weight_ratio`: `Ht` divided by `Wt` (with zero values in `Wt` replaced by NaN to avoid division errors).
  - `num_missing_row`: count of missing values per row.
- Splits categorical columns into low-cardinality (10 or fewer unique values) and high-cardinality groups, applying a different encoding strategy to each.
- Applies frequency encoding to high-cardinality categorical columns (each category is replaced by its frequency in the training set).
- Builds a `ColumnTransformer` pipeline that imputes missing values (median for numeric, constant fill for categorical), scales numeric features, and one-hot encodes low-cardinality categorical features.
- Runs a preliminary `RandomForestClassifier` to rank features by importance and selects the top 50 features for downstream modeling.
- Trains and tunes three classifiers with `GridSearchCV` and stratified 5-fold cross-validation, using F1-score as the scoring metric:
  - `RandomForestClassifier`
  - `LogisticRegression` (with `saga` solver, L1/L2 penalty)
  - `LGBMClassifier` (LightGBM)
- Evaluates each tuned model on a held-out test set with accuracy, F1-score, ROC AUC, confusion matrix, and a full classification report.

## Project Structure

This project consists of a single Jupyter notebook, organized into 14 numbered sections marked by markdown headers:

- `Prudential_Life_Insurance_Assessment.ipynb`: the full pipeline, structured as follows.
  1. **Ucitavanje podataka** (Data loading): reads `train.csv` and previews shape, info, and target distribution.
  2. **Brza analiza NA vrednosti** (Missing value overview): bar chart of the columns with the most missing values.
  3. **Priprema ciljne promenljive** (Target preparation): builds the binary `Response_bin` target.
  4. **Drop 10 najređih kolona** (Dropping rare columns): removes a predefined list of categorical columns per the given instructions.
  5. **Konstrukcija novih atributa** (Feature engineering): builds the summary and interaction features described above.
  6. **Podela podataka** (Train/test split): stratified 80/20 split by `Response_bin`.
  7. **Identifikacija numeričkih i kategorijskih kolona** (Column typing): separates numeric and categorical columns after feature engineering.
  8. **Razdvajanje kategorija po kardinalnosti** (Cardinality split and frequency encoding): splits categorical columns into low and high cardinality, and frequency-encodes the high-cardinality group.
  9. **Sastavljanje listi kolona** (Assembling column lists): finalizes the numeric and low-cardinality column lists for the `ColumnTransformer`.
  10. **ColumnTransformer** (Preprocessing pipeline): imputation, scaling, and one-hot encoding.
  11. **Feature selection**: RandomForest-based importance ranking and top-50 feature selection.
  12. **Modeli** (Modeling): GridSearchCV tuning for RandomForest, LogisticRegression, and LGBMClassifier.
  13. **Evaluacija modela** (Model evaluation): metrics and reports for all three tuned models on the test set.
  14. **Kratki zaključci i šta dalje** (Conclusions and next steps): printed summary of findings and suggested improvements.

Note that all section headers and inline comments in the notebook are written in Serbian; this README describes their content in English.

## Installation

The notebook requires Python 3 and the following packages. LightGBM is not preinstalled in all environments, so the first cell installs it directly:

```bash
pip install lightgbm
pip install pandas numpy matplotlib seaborn scikit-learn
```

You will also need `train.csv` (the Prudential Life Insurance Assessment dataset) placed in the same directory as the notebook, since it is loaded with a relative path.

## Usage

Open the notebook and run all cells in order, since later sections depend on variables created earlier (for example, the `ColumnTransformer` in section 10 depends on the column lists built in sections 7 to 9):

```bash
jupyter notebook Prudential_Life_Insurance_Assessment.ipynb
```

Key snippets from the workflow:

Loading the data (only the first 59,000 rows are used):

```python
df = pd.read_csv('train.csv').iloc[:59000].copy()
```

Building the binary target:

```python
df['Response_bin'] = df['Response'].apply(lambda x: 0 if x <= 4 else 1)
```

Running the preprocessing pipeline:

```python
X_train_pre = preprocessor.fit_transform(X_train_enc)
X_test_pre  = preprocessor.transform(X_test_enc)
```

Selecting the top features by RandomForest importance:

```python
top_k = min(50, len(feat_imp))
selected_feats = feat_imp.head(top_k).index.tolist()
```

Evaluating a trained model:

```python
evaluate_model("RandomForest (best from GridSearch)", y_test, y_pred_rf, y_proba_rf)
```

## Configuration

There are no external configuration files or environment variables. All parameters are set directly in code, including:

- `test_size=0.2` and `random_state=42` for the train/test split.
- `top_n = 13` for the number of columns shown in the missing-value bar chart.
- `top_k = 50` for the number of features kept after RandomForest-based selection.
- Hyperparameter grids for each model (`param_grid_rf`, `param_grid_log`, `param_grid_lgb`), each searched with 5-fold stratified cross-validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`).

If you want to change the dataset size, split ratio, or number of selected features, edit these values directly in the corresponding cells.

## Technologies Used

- **Language**: Python 3
- **Data handling**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine learning**: scikit-learn (preprocessing, pipelines, model selection, metrics, RandomForestClassifier, LogisticRegression)
- **Gradient boosting**: LightGBM (`LGBMClassifier`)

## Author

*M. E. Petra Bošković*
