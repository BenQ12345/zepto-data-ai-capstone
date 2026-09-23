# Model comparison and recommendation

| model_type     | model                          |   accuracy |   precision |     recall |         f1 |        auc |   Regression MAE |   Regression RMSE |   Regression R2 |   Regression Adjusted R2 |
|:---------------|:-------------------------------|-----------:|------------:|-----------:|-----------:|-----------:|-----------------:|------------------:|----------------:|-------------------------:|
| classification | Logistic Regression            |   0.804469 |    0.793103 |   0.666667 |   0.724409 |   0.843742 |          nan     |          nan      |      nan        |               nan        |
| classification | Decision Tree                  |   0.765363 |    0.754717 |   0.57971  |   0.655738 |   0.797101 |          nan     |          nan      |      nan        |               nan        |
| classification | Random Forest                  |   0.810056 |    0.79661  |   0.681159 |   0.734375 |   0.828722 |          nan     |          nan      |      nan        |               nan        |
| regression     | Multivariate Linear Regression | nan        |  nan        | nan        | nan        | nan        |           18.077 |           27.7237 |        0.503302 |                 0.422142 |

## Deployment recommendation
For the fixed stratified test split, **Random Forest** is the classifier selected for deployment because it has the highest test-set F1 among the three compared classifiers (F1 = 0.734) while also achieving accuracy = 0.810, recall = 0.681, precision = 0.797, and AUC = 0.829. These metrics are considered together because the target is a classification problem and the business may care about both false positives and false negatives. The regression metrics are reported separately because MAE, RMSE, R² and adjusted R² describe a different continuous prediction task and are not directly comparable to classification scores. The complete fitted preprocessing + classifier pipeline is saved as `titanic_best_pipeline.joblib` and reload-tested on raw, unpreprocessed feature data.
