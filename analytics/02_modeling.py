from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

BASE_DIR = Path(__file__).resolve().parent
CHART_DIR = BASE_DIR / "charts"
RESULT_DIR = BASE_DIR / "results"
CHART_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)

# IMPORTANT: modeling continues from the exact titanic.csv produced by 01_eda.py.
# There is intentionally no second sns.load_dataset(...) call here.
df = pd.read_csv(BASE_DIR / "titanic.csv")

TARGET = "survived"
FEATURES = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
NUMERIC = ["pclass", "age", "sibsp", "parch", "fare"]
CATEGORICAL = ["sex", "embarked"]
X = df[FEATURES].copy()
y = df[TARGET].astype(int).copy()

class_counts = y.value_counts().sort_index()
class_balance = pd.DataFrame({"class": [0, 1], "count": [class_counts.get(0, 0), class_counts.get(1, 0)]})
class_balance["pct"] = class_balance["count"] / len(y) * 100
print("=== CLASS BALANCE ===")
print(class_balance.to_string(index=False))
class_balance.to_csv(RESULT_DIR / "class_balance.csv", index=False)

# Stratify before ANY model preprocessing. This preserves the observed target proportion.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            NUMERIC,
        ),
        (
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            CATEGORICAL,
        ),
    ],
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
}

fitted = {}
metrics_rows = []
roc_data = {}

for name, estimator in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    fitted[name] = pipe
    roc_data[name] = roc_curve(y_test, proba)
    row = {
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "auc": auc,
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    metrics_rows.append(row)
    print(f"\n=== {name} ===")
    print(confusion_matrix(y_test, pred))
    print({k: v for k, v in row.items() if k != "confusion_matrix"})

# Decision-tree visualization with post-transform feature names.
tree_pipe = fitted["Decision Tree"]
pre = tree_pipe.named_steps["preprocessor"]
tree_model = tree_pipe.named_steps["model"]
feature_names = list(pre.named_transformers_["num"].get_feature_names_out(NUMERIC)) + list(
    pre.named_transformers_["cat"].get_feature_names_out(CATEGORICAL)
)
fig, ax = plt.subplots(figsize=(22, 12))
plot_tree(tree_model, feature_names=feature_names, class_names=["not_survived", "survived"], filled=False, max_depth=4, ax=ax)
ax.set_title("Decision tree (first four levels shown)")
fig.tight_layout()
fig.savefig(CHART_DIR / "decision_tree.png", dpi=140)
plt.close(fig)

# ROC curves + AUC.
fig, ax = plt.subplots(figsize=(8, 6))
for name, (fpr, tpr, _) in roc_data.items():
    auc = next(r["auc"] for r in metrics_rows if r["model"] == name)
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves")
ax.legend()
fig.tight_layout()
fig.savefig(CHART_DIR / "roc_curves.png", dpi=160)
plt.close(fig)

metrics_df = pd.DataFrame(metrics_rows)
metrics_df.to_csv(RESULT_DIR / "classifier_metrics.csv", index=False)

# Imbalance comparison on one classifier: Logistic Regression baseline, class_weight, SMOTE.
imbalance_rows = []
for label, clf in [
    ("baseline", LogisticRegression(max_iter=2000, random_state=42)),
    ("class_weight_balanced", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
]:
    pipe = Pipeline([("preprocessor", preprocessor), ("model", clf)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    imbalance_rows.append({
        "strategy": label,
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
    })

smote_pipe = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", LogisticRegression(max_iter=2000, random_state=42)),
])
smote_pipe.fit(X_train, y_train)
pred_smote = smote_pipe.predict(X_test)
imbalance_rows.append({
    "strategy": "SMOTE_train_only",
    "precision": precision_score(y_test, pred_smote, zero_division=0),
    "recall": recall_score(y_test, pred_smote, zero_division=0),
    "f1": f1_score(y_test, pred_smote, zero_division=0),
})
imbalance_df = pd.DataFrame(imbalance_rows)
imbalance_df.to_csv(RESULT_DIR / "imbalance_comparison.csv", index=False)
print("\n=== IMBALANCE COMPARISON ===")
print(imbalance_df.to_string(index=False))

# GridSearchCV for Random Forest with oob_score=True as required.
rf_base = RandomForestClassifier(oob_score=True, random_state=42, bootstrap=True)
rf_pipe = Pipeline([("preprocessor", preprocessor), ("model", rf_base)])
param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 5, 10],
    "model__max_features": ["sqrt", "log2"],
}
grid = GridSearchCV(rf_pipe, param_grid=param_grid, scoring="f1", cv=5, n_jobs=-1, refit=True)
grid.fit(X_train, y_train)
best_rf = grid.best_estimator_
best_rf_model = best_rf.named_steps["model"]
print("\n=== GRID SEARCH ===")
print("Best parameters:", grid.best_params_)
print("Best CV F1:", grid.best_score_)
print("OOB score:", best_rf_model.oob_score_)
with open(RESULT_DIR / "grid_search.txt", "w", encoding="utf-8") as f:
    f.write(f"Best parameters: {grid.best_params_}\n")
    f.write(f"Best CV F1: {grid.best_score_:.6f}\n")
    f.write(f"OOB score: {best_rf_model.oob_score_:.6f}\n")

# Regression side-task: predict fare from the other available features.
reg_features = [c for c in df.columns if c not in ["fare"] and df[c].dtype != "datetime64[ns]"]
# Exclude text-like identifiers and duplicated outcome labels that do not serve as useful predictors.
reg_features = [c for c in reg_features if c not in ["alive", "class"]]
Xr = df[reg_features].copy()
yr = df["fare"].copy()
num_reg = Xr.select_dtypes(include=[np.number, "bool"]).columns.tolist()
cat_reg = [c for c in Xr.columns if c not in num_reg]
reg_pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_reg),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_reg),
])
Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.20, random_state=42)
reg_pipe = Pipeline([("preprocessor", reg_pre), ("model", LinearRegression())])
reg_pipe.fit(Xr_train, yr_train)
reg_pred = reg_pipe.predict(Xr_test)
mae = mean_absolute_error(yr_test, reg_pred)
rmse = float(np.sqrt(mean_squared_error(yr_test, reg_pred)))
r2 = r2_score(yr_test, reg_pred)
n = len(yr_test)
p = len(reg_pipe.named_steps["preprocessor"].get_feature_names_out())
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
resid = yr_test - reg_pred
fitted_vals = pd.Series(reg_pred, index=yr_test.index)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(fitted_vals, resid, alpha=0.65)
ax.axhline(0, linestyle="--", linewidth=1)
ax.set_xlabel("Fitted fare")
ax.set_ylabel("Residual")
ax.set_title("Linear regression residual plot")
fig.tight_layout()
fig.savefig(CHART_DIR / "regression_residuals.png", dpi=160)
plt.close(fig)

# A simple textual heteroscedasticity check: compare residual spread in fitted-value halves.
median_fit = fitted_vals.median()
left_std = resid[fitted_vals <= median_fit].std()
right_std = resid[fitted_vals > median_fit].std()
ratio = max(left_std, right_std) / max(min(left_std, right_std), 1e-12)
hetero = "evidence of heteroscedasticity" if ratio >= 1.5 else "no strong evidence of heteroscedasticity"
print("\n=== REGRESSION ===")
print({"MAE": mae, "RMSE": rmse, "R2": r2, "Adjusted_R2": adj_r2, "residual_spread_ratio": ratio, "conclusion": hetero})
reg_metrics = {"MAE": mae, "RMSE": rmse, "R2": r2, "Adjusted_R2": adj_r2, "Residual spread ratio": ratio, "Heteroscedasticity conclusion": hetero}
with open(RESULT_DIR / "regression_metrics.json", "w", encoding="utf-8") as f:
    json.dump(reg_metrics, f, indent=2)

# Final deployment candidate: choose the best classifier by F1, then AUC, without averaging unlike metrics.
selected = sorted(metrics_rows, key=lambda r: (r["f1"], r["auc"], r["recall"]), reverse=True)[0]
selected_name = selected["model"]
final_pipeline = fitted[selected_name]
joblib.dump(final_pipeline, BASE_DIR / "titanic_best_pipeline.joblib")
reloaded = joblib.load(BASE_DIR / "titanic_best_pipeline.joblib")
raw_sample = X_test.iloc[[0]]
assert reloaded.predict(raw_sample).shape == (1,)

# Comparison table: classification and regression are separate metric groups.
classification_table = metrics_df[["model", "accuracy", "precision", "recall", "f1", "auc"]].copy()
classification_table["model_type"] = "classification"
classification_table["Regression MAE"] = np.nan
classification_table["Regression RMSE"] = np.nan
classification_table["Regression R2"] = np.nan
classification_table["Regression Adjusted R2"] = np.nan
regression_row = pd.DataFrame([{
    "model": "Multivariate Linear Regression",
    "model_type": "regression",
    "accuracy": np.nan, "precision": np.nan, "recall": np.nan, "f1": np.nan, "auc": np.nan,
    "Regression MAE": mae, "Regression RMSE": rmse, "Regression R2": r2, "Regression Adjusted R2": adj_r2,
}])
comparison = pd.concat([classification_table, regression_row], ignore_index=True)
comparison = comparison[["model_type", "model", "accuracy", "precision", "recall", "f1", "auc", "Regression MAE", "Regression RMSE", "Regression R2", "Regression Adjusted R2"]]
comparison.to_csv(RESULT_DIR / "model_comparison.csv", index=False)

# Required recommendation in text, using the best F1 classifier from this fixed split.
with open(RESULT_DIR / "model_recommendation.md", "w", encoding="utf-8") as f:
    f.write("# Model comparison and recommendation\n\n")
    f.write(comparison.to_markdown(index=False))
    f.write("\n\n## Deployment recommendation\n")
    f.write(
        f"For the fixed stratified test split, **{selected_name}** is the classifier selected for deployment because it has the highest test-set F1 among the three compared classifiers (F1 = {selected['f1']:.3f}) while also achieving accuracy = {selected['accuracy']:.3f}, recall = {selected['recall']:.3f}, precision = {selected['precision']:.3f}, and AUC = {selected['auc']:.3f}. These metrics are considered together because the target is a classification problem and the business may care about both false positives and false negatives. The regression metrics are reported separately because MAE, RMSE, R² and adjusted R² describe a different continuous prediction task and are not directly comparable to classification scores. The complete fitted preprocessing + classifier pipeline is saved as `titanic_best_pipeline.joblib` and reload-tested on raw, unpreprocessed feature data.\n"
    )

print(f"\nSaved complete fitted pipeline: {BASE_DIR / 'titanic_best_pipeline.joblib'}")
print(f"Selected classifier by test F1: {selected_name}")
