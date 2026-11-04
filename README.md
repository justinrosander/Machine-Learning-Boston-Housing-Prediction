# Boston Housing Price Prediction

Predict median owner-occupied home values from locality features, and identify which features move price. The work is a supervised linear-regression analysis of the classic Boston housing data (1970 Boston SMSA): 506 tracts, 13 columns, no missing values, and no duplicate rows.

The full analysis, plots, and model diagnostics are in [`Learners_Notebook_Boston_house_price_Full Code.ipynb`](Learners_Notebook_Boston_house_price_Full%20Code.ipynb). A rendered copy is at [justinrosander.github.io/Machine-Learning-Boston-Housing-Prediction](https://justinrosander.github.io/Machine-Learning-Boston-Housing-Prediction).

## Outline

1. Load `data/raw/Boston.csv` and audit quality on the raw and processed tables.
2. Explore the target and each predictor (shape, skew, outliers).
3. Map pairwise relationships with `MEDV` and among predictors.
4. Run multivariate diagnostics (VIF, partial regression, residuals, Mahalanobis distance).
5. Preprocess for a linear model: log the target, drop redundant collinear features, cap selected outliers, and scale predictors.
6. Fit OLS, prune weak predictors, and compare the final specification with Ridge and Lasso.
7. Check linear-regression assumptions and save the final model.

| Role | Columns |
| --- | --- |
| Target | `MEDV` — median home value, in $1,000s (observed range 5 to 50, capped at 50) |
| Predictors | `CRIM`, `ZN`, `INDUS`, `CHAS`, `NOX`, `RM`, `AGE`, `DIS`, `RAD`, `TAX`, `PTRATIO`, `LSTAT` |

## EDA highlights

`MEDV` is right-skewed (skewness 1.108). The mean (22.53) sits above the median (21.2). Most tracts fall between the quartiles 17.03 and 25.00, with a thin high-price tail and a hard ceiling at 50.

The strongest links to price are socioeconomic status and dwelling size:

| Feature | Correlation with `MEDV` | Direction |
| --- | ---: | --- |
| `LSTAT` | −0.738 | Higher share of lower-status population, lower value |
| `RM` | +0.695 | More rooms per dwelling, higher value |
| `PTRATIO` | −0.508 | Higher pupil–teacher ratio, lower value |
| `INDUS` | −0.484 | More non-retail business land, lower value |
| `TAX` | −0.469 | Higher property-tax rate, lower value |
| `NOX` | −0.427 | Higher nitric-oxide concentration, lower value |

Several predictors are far from symmetric. `CRIM` is heavily right-skewed (skewness 5.22), `ZN` is mostly zeros (about 74%) with a long right tail, and `CHAS` is rare (about 7% of tracts border the Charles River). `LSTAT` and `DIS` are also right-skewed. Those shapes supported a log target and selective transforms, with scaling reserved for the model step.

Predictors also overlap. Eight pairs meet `|r| >= 0.7`. The tightest is `RAD`–`TAX` (0.910), then `NOX`–`DIS` (−0.769), `INDUS`–`NOX` (0.764), and `AGE`–`DIS` (−0.748). Early VIF on the full engineered set was high (for example `RM_NOX_RATIO` 57.5, `LOG_CRIM` 30.0, `RAD` 27.9), so raw highway access, tax, and their near-duplicate logs were dropped before the final fit.

Outliers were capped on a short list (`ZN`, `CRIM`, `CHAS`, `RM`, `TAX_RAD_RATIO`) so extreme values lost leverage while all 506 rows stayed in the sample. A Mahalanobis screen flagged 58 tracts as unusual in combined predictor space. Those points were reviewed as multivariate outliers, not deleted by default.

Log1p of `MEDV` cut skewness from 1.108 to −0.241, which is why the models below predict `LOG_MEDV`.

## Visualizations

All figures live in the notebook.

**Univariate**

- `MEDV` histogram with KDE, paired with a boxplot that shows the upper cap near 50.
- Histogram-and-KDE grid for every predictor.
- Boxplot grids for predictor outliers, including a grid that puts `MEDV` back in with the features.
- Skewness table used to prioritize transforms.

**Bivariate**

- Scatter plots with a fitted line for `MEDV` against the six strongest correlates (`LSTAT`, `RM`, `PTRATIO`, `INDUS`, `TAX`, `NOX`).
- Scatter grid of every pair with `|r| >= 0.7`, annotated with the correlation.
- Boxplots of `MEDV` by binary or grouped features such as the Charles River indicator.
- Corner pair plot of `MEDV`, `LSTAT`, `RM`, `PTRATIO`, `TAX`, and `NOX`.

**Multivariate and model diagnostics**

- Horizontal bar chart of the largest VIF values.
- Added-variable (partial regression) grid for the strongest joint predictors.
- Residuals versus fitted values, plus residuals versus individual predictors.
- Standardized coefficient bars (direction and relative size on a common scale).
- Five-fold cross-validation boxplots of R², MAE, and RMSE.
- Joint density of `RM` and `LSTAT` as an interaction cue.
- Mahalanobis distance histogram and a sorted-distance scatter plot.

## Model highlights

The modeling frame is an 80/20 train–test split (`random_state=42`): 404 training rows and 102 test rows. Predictors are standardized. The target is `LOG_MEDV = log1p(MEDV)`.

An initial OLS model on the reduced feature set already generalized well (train R² 0.803, test R² 0.787) and had no VIF above 5. Pruning then dropped `INDUS`, `ZN`, `AGE`, and `CHAS` for weak partial association. The retained model uses nine features:

`CRIM`, `NOX`, `RM`, `DIS`, `PTRATIO`, `LSTAT`, `MEDV_CAP50`, `HIGH_CRIME`, `TAX_RAD_RATIO`

Metrics below are on the log1p scale. Dollar prices are `1000 * expm1(prediction)`.

| Metric | Train | Test |
| --- | ---: | ---: |
| R² | 0.801 | 0.790 |
| Adjusted R² | 0.797 | 0.769 |
| RMSE | 0.174 | 0.171 |
| MAE | 0.130 | 0.118 |
| MAPE | 4.42% | 4.06% |

Five-fold CV on the training design is stable: R² 0.778 ± 0.048 and RMSE 0.181 ± 0.015. Train and test stay close, so the fit is not memorizing the training tracts.

Standardized coefficients, largest absolute effect first:

| Feature | Coefficient | Reading |
| --- | ---: | --- |
| `LSTAT` | −0.168 | Lower-status share lowers predicted value |
| `CRIM` | −0.113 | Higher crime rate lowers predicted value |
| `MEDV_CAP50` | +0.080 | Tracts at the $50k cap sit higher |
| `RM` | +0.074 | More rooms raise predicted value |
| `NOX` | −0.071 | Higher pollution lowers predicted value |
| `PTRATIO` | −0.067 | Higher pupil–teacher ratio lowers predicted value |
| `DIS` | −0.063 | Greater distance to employment centers lowers predicted value |
| `TAX_RAD_RATIO` | −0.041 | Tax relative to highway access lowers predicted value |
| `HIGH_CRIME` | +0.025 | Smallest effect; partial association is weak (p ≈ 0.07) |

RidgeCV and LassoCV on the same nine scaled features match OLS (test R² 0.789 and 0.790). Lasso did not zero any coefficient. After pruning, extra shrinkage adds little.

Assumption checks on the pre-prune specification: linearity and independence look acceptable (Durbin–Watson 2.13), and VIF stays under 5. Homoscedasticity (Breusch–Pagan and White) and residual normality (Shapiro–Wilk and Anderson–Darling) do not. Coefficients are still useful for direction and rank. Interval estimates would need robust standard errors such as HC3.

Saved artifacts:

- `models/final_linear_regression_model.joblib`
- `models/final_model_metadata.json`
- `models/final_selected_features.json`

## Takeaways

- Price in this sample is driven first by `LSTAT` and `CRIM` (down) and by rooms and the censored top of the market (up). Pollution, school crowding, and distance to jobs move the same direction as crime, with smaller standardized effects.
- `MEDV_CAP50` and `TAX_RAD_RATIO` earned a place in the final set. The ceiling at 50 is real structure, and the tax-to-highway ratio keeps a slice of the `RAD`–`TAX` signal without keeping both raw columns.
- A nine-feature OLS model explains about 79% of log-price variance on held-out tracts, with about 4% MAPE on that log scale. Ridge and Lasso do not improve it.
- The data are a single 1970 Boston-area cross-section. They support a clear baseline and an interpretable story. They do not support claims about today’s markets or other cities. The cap at 50 also compresses the high end.
- Useful next steps are robust standard errors, a check of the 29 high-Cook’s-distance points, a few targeted interactions (rooms with `LSTAT`, for example), and a true external holdout from another place or year.

## Repository layout

```text
data/raw/Boston.csv                  original table
data/processed/                      modeling and hypothesis-testing tables
models/                              final OLS pipeline and metadata
Learners_Notebook_Boston_house_price_Full Code.ipynb
reports/                             written summaries and exported figures
src/                                 reusable helpers
```

The notebook reads `data/raw/Boston.csv` and `data/processed/Boston_prepared_for_hypothesis_testing.csv`. Libraries used there are pandas, NumPy, matplotlib, seaborn, scikit-learn, statsmodels, SciPy, and joblib.
