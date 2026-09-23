<<<<<<< HEAD
# Analytics Pipeline

## Execution order

```bash
python 01_eda.py
python 02_modeling.py
```

`01_eda.py` is the **only** place that calls `sns.load_dataset("titanic")`. It immediately snapshots the raw frame to `titanic.csv`, then profiles and cleans an EDA copy. `02_modeling.py` reads that same `titanic.csv`; it does not perform a second network/cache load. An offline fallback in `01_eda.py` only activates when the committed `titanic.csv` already exists and the environment cannot reach Seaborn's dataset repository.

## Profiling and missing-value decisions

The raw snapshot is **891 rows × 15 columns**. The affected columns and measured missing percentages were:

| column      |   missing_pct | strategy                                                                                               |
|:------------|--------------:|:-------------------------------------------------------------------------------------------------------|
| deck        |     77.2166   | deck: 77.22% missing (>30%), dropped because imputation would be unreliable at this missingness level. |
| age         |     19.8653   | age: 19.87% missing (5%-30%), median-imputed with 28.00.                                               |
| embarked    |      0.224467 | embarked: 0.22% missing (<5%), dropped affected rows.                                                  |
| embark_town |      0.224467 | embark_town: 0.22% missing (<5%), dropped affected rows.                                               |

The threshold rule is applied literally: under 5% → drop affected rows; 5–30% → impute; above 30% → explicitly drop the column because imputation is unreliable at that missingness level. Therefore `age` (19.87%) is median-imputed at 28.00 for EDA, `embarked` and `embark_town` (0.22% each) have their affected rows dropped, and `deck` (77.22%) is dropped. `who` has no missing values in the Seaborn dataset and therefore requires no missing-value strategy.

## Univariate results

The EDA-generated charts are under `charts/`.

- **Age:** IQR bounds are 2.50 to 54.50, with **65** outliers after the EDA cleaning step.
- **Fare:** IQR bounds are -26.76 to 65.66, with **114** outliers.
- Fare mean = **32.10**, median = **14.45**, mode = **8.05**. Because mean > median > mode, the fare distribution is **right-skewed**.

## Bivariate survival results

### Survival by sex

- Female: **0.740**
- Male: **0.189**

### Survival by passenger class

- Pclass 1: **0.626**
- Pclass 2: **0.473**
- Pclass 3: **0.242**

### Survival by sex × class

The actual sex × class table is produced and printed by `01_eda.py`; the values are:

| sex | pclass | survival_rate |
|---|---:|---:|
| female | 1 | 0.967 |
| female | 2 | 0.921 |
| female | 3 | 0.500 |
| male | 1 | 0.369 |
| male | 2 | 0.157 |
| male | 3 | 0.135 |

The sex × class calculation uses explicit boolean masks with `&`, as required.

## Correlation story

The heatmap is `charts/correlation_heatmap.png`. The matrix is computed on **exactly** `survived`, `pclass`, `age`, `sibsp`, `parch`, and `fare`; `adult_male` and `alone` are excluded.

The two strongest absolute off-diagonal correlations are:

1. **pclass vs fare:** r = **-0.548**. Higher passenger class number is associated with lower fare, reflecting the expected socioeconomic structure of the ticket classes.
2. **sibsp vs parch:** r = **0.415**. Sibling/spouse counts and parent/child counts move together because both are family-travel measures.

## Multivariate data story

### 1. Survival by sex — `charts/survival_by_sex.png`
The chart shows a large separation in survival rates between female and male passengers. This supports the data story that sex was strongly associated with observed survival, although it describes an association rather than proving causality.

### 2. Survival by class — `charts/survival_by_class.png`
Survival rates differ materially across passenger classes, with higher-class passengers having higher survival rates in this dataset. Passenger class therefore adds information beyond sex alone and is a useful predictor candidate.

### 3. Survival by sex and class — `charts/survival_sex_class.png`
Combining sex with passenger class reveals that the survival gap is not identical within every class. The interaction-like pattern suggests that considering the two attributes together explains more of the observed outcome structure than viewing either variable in isolation.

### 4. Age, fare and survival — `charts/age_fare_survival.png`
The scatter plot shows that survival outcomes occupy different regions of age and fare space, with fare partly separating passengers into different economic segments. There is substantial overlap, so these variables should be used together with categorical and family-travel features rather than treated as deterministic rules.

### 5. Fare by class and survival — `charts/fare_class_survival.png`
Fare distributions are strongly shaped by passenger class, and survivors are distributed differently within those class-specific fare ranges. This indicates that fare contains both its own signal and information correlated with class.

## EDA-only standardization check

`results/standardization_check.csv` confirms approximately zero means and unit population standard deviations after StandardScaler:

- Age: mean ≈ **0.0000**, std(ddof=0) = **1.0000**
- Fare: mean ≈ **0.0000**, std(ddof=0) = **1.0000**

These transformed columns are diagnostic only. The modeling pipeline fits its own scaler on the training split to prevent test leakage.

## Modeling design

Observed target balance:

|   class |   count |     pct |
|--------:|--------:|--------:|
|       0 |     549 | 61.6162 |
|       1 |     342 | 38.3838 |

The split is performed with `stratify=y` before any model preprocessing. Stratification keeps the train/test survived/not-survived proportions close to the observed class balance, which makes evaluation more representative and avoids accidentally creating a test set with an unusual class mix.

Numeric model features use training-only median imputation plus `StandardScaler`. `sex` and `embarked` use training-only most-frequent imputation plus `OneHotEncoder(handle_unknown="ignore")`. These operations live inside a `ColumnTransformer` within a scikit-learn `Pipeline`, so the imputer, encoder and scaler are fitted through `fit(X_train, y_train)` only and merely transform the test rows.

### Three-classifier comparison

| model               |   accuracy |   precision |   recall |       f1 |      auc |
|:--------------------|-----------:|------------:|---------:|---------:|---------:|
| Logistic Regression |   0.804469 |    0.793103 | 0.666667 | 0.724409 | 0.843742 |
| Decision Tree       |   0.765363 |    0.754717 | 0.57971  | 0.655738 | 0.797101 |
| Random Forest       |   0.810056 |    0.79661  | 0.681159 | 0.734375 | 0.828722 |

The same stratified split is used for all three classifiers. Each model also has a confusion matrix stored through the script's printed output, and the ROC curves are in `charts/roc_curves.png`. The Decision Tree is rendered with labeled transformed features and class names in `charts/decision_tree.png`.

## Imbalance comparison

| strategy              |   precision |   recall |       f1 |
|:----------------------|------------:|---------:|---------:|
| baseline              |    0.793103 | 0.666667 | 0.724409 |
| class_weight_balanced |    0.72973  | 0.782609 | 0.755245 |
| SMOTE_train_only      |    0.739726 | 0.782609 | 0.760563 |

On this split, the baseline emphasizes precision, while both balancing strategies increase recall. SMOTE gives the highest F1 among the three imbalance variants in this experiment (**0.761**) and improves recall to **0.783**, showing the benefit of training-fold oversampling for this class balance. SMOTE is implemented inside an imbalanced-learn pipeline, so it is applied only to training data.

## Random Forest tuning and OOB

- Best parameters: **max_depth = 5, max_features = `sqrt`, n_estimators = 200**
- Best 5-fold CV F1: **0.744**
- OOB score from the refitted `RandomForestClassifier(oob_score=True, ...)`: **0.830**

## Regression side-task

The multivariate linear regression predicts `fare` from the other available features after excluding the exact target `fare` and duplicate target labels `alive`/`class`. The test metrics are:

| Metric | Value |
|---|---:|
| MAE | 18.077 |
| RMSE | 27.724 |
| R² | 0.503 |
| Adjusted R² | 0.422 |

The residual-spread ratio is **3.500**, and the result is interpreted as **evidence of heteroscedasticity** because residual variance changes materially across the fitted-value range. The residual plot is `charts/regression_residuals.png`.

## Final comparison table

Classification and regression are different prediction tasks, so their metrics are kept in separate metric columns and are not ranked against one another numerically.

| model_type     | model                          |   accuracy |   precision |     recall |         f1 |        auc |   Regression MAE |   Regression RMSE |   Regression R2 |   Regression Adjusted R2 |
|:---------------|:-------------------------------|-----------:|------------:|-----------:|-----------:|-----------:|-----------------:|------------------:|----------------:|-------------------------:|
| classification | Logistic Regression            |   0.804469 |    0.793103 |   0.666667 |   0.724409 |   0.843742 |          nan     |          nan      |      nan        |               nan        |
| classification | Decision Tree                  |   0.765363 |    0.754717 |   0.57971  |   0.655738 |   0.797101 |          nan     |          nan      |      nan        |               nan        |
| classification | Random Forest                  |   0.810056 |    0.79661  |   0.681159 |   0.734375 |   0.828722 |          nan     |          nan      |      nan        |               nan        |
| regression     | Multivariate Linear Regression | nan        |  nan        | nan        | nan        | nan        |           18.077 |           27.7237 |        0.503302 |                 0.422142 |

## Final classifier recommendation

Among the three baseline classifiers on the fixed stratified test split, **Random Forest** is selected for the deployment artifact because it has the highest test F1 (**0.734**) and the highest accuracy (**0.810**) of the three, with precision **0.797**, recall **0.681**, and AUC **0.829**. Logistic Regression has a slightly higher AUC (**0.844**) but lower F1 (**0.724**), while Decision Tree trails on both F1 (**0.656**) and AUC (**0.797**) in this run. These metrics are considered together because classification errors have different implications than continuous fare-prediction errors. The complete preprocessing + classifier pipeline is saved as `titanic_best_pipeline.joblib` and reload-tested on raw, unpreprocessed feature data.

## Saved artifacts

- `titanic.csv` — offline raw snapshot committed in `/analytics`.
- `titanic_best_pipeline.joblib` — complete fitted preprocessing + classifier pipeline.
- `charts/` — supporting chart artifacts.
- `results/` — machine-readable metric outputs and generated evidence.
=======
# Module 2 - Analytics

This folder is reserved for the analytics module from the capstone specification.
>>>>>>> a4a6ce65e9b1429e1138e7b9893d1b12e7f926f9
