# EDA Report

## Missing-value decisions
- deck: 77.22% missing (>30%), dropped because imputation would be unreliable at this missingness level.
- age: 19.87% missing (5%-30%), median-imputed with 28.00.
- embarked: 0.22% missing (<5%), dropped affected rows.
- embark_town: 0.22% missing (<5%), dropped affected rows.

## Outliers and fare shape
- Age IQR outliers: **65** using bounds [2.50, 54.50].
- Fare IQR outliers: **114** using bounds [-26.76, 65.66].
- Fare mean = 32.10, median = 14.45, mode = 8.05. The ordering indicates a **right-skewed** distribution because the mean is above the median and the median is above the mode.

## Survival breakdowns

### Sex
- female: 0.740
- male: 0.189

### Passenger class
- class 1: 0.626
- class 2: 0.473
- class 3: 0.242

### Sex × passenger class

| sex    |   pclass |   survival_rate |
|:-------|---------:|----------------:|
| female |        1 |        0.967391 |
| female |        2 |        0.921053 |
| female |        3 |        0.5      |
| male   |        1 |        0.368852 |
| male   |        2 |        0.157407 |
| male   |        3 |        0.135447 |

## Two strongest correlations
- **pclass vs fare**: r = -0.548. The absolute coefficient is 0.548, so this is one of the two strongest relationships in the required six-feature matrix.
- **sibsp vs parch**: r = 0.415. The absolute coefficient is 0.415, so this is one of the two strongest relationships in the required six-feature matrix.

## Multivariate chart interpretations
### Survival by sex
The sex chart shows a large separation in survival rates between female and male passengers. This supports the data story that sex was strongly associated with survival, although it describes an association rather than proving causality.

### Survival by class
Survival rates differ materially across passenger classes, with higher-class passengers having higher survival rates in this dataset. Passenger class therefore adds information beyond sex alone and is a useful predictor candidate.

### Survival by sex and class
Combining sex with passenger class reveals that the survival gap is not identical within every class. The interaction-like pattern suggests that considering the two attributes together explains more of the observed outcome structure than viewing either variable in isolation.

### Age, fare and survival
The scatter plot shows that survival outcomes occupy different regions of age and fare space, with fare partly separating passengers into different economic segments. There is substantial overlap, so these variables should be used together with categorical and family-travel features rather than treated as deterministic rules.

### Fare by class and survival
Fare distributions are strongly shaped by passenger class, and survivors are distributed differently within those class-specific fare ranges. This indicates that fare contains both its own signal and information correlated with class.

## Standardization
The EDA-only StandardScaler check produces means approximately equal to 0 and population standard deviations approximately equal to 1 for age and fare. These transformed columns are diagnostic only; the modeling module fits a separate scaler on the training split to prevent test leakage.
