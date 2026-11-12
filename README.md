# Environmental-ML-Project
Project 2 as part of the course 2107541: Machine Learning for Environmental Application, where ML project was done to estimate PM2.5 of Beijing, China


# Machine Learning for Daily PM2.5 Forecasting and Interpretation of Air Pollution Variability in Beijing

## 1. Environmental Context and Study Rationale

Fine particulate matter (PM2.5), comprising particles with diameters of 2.5 µm or less, can penetrate deeply into the respiratory system and is associated with cardiovascular and respiratory disease. During 2010–2014, the daily mean PM2.5 concentration in the Beijing dataset exceeded the Chinese 24-hour standard of 75 µg/m³ on 52% of days and the World Health Organization's 2021 guideline of 15 µg/m³ on 96% of days. The latter is used as a retrospective benchmark. Severe, multi-day pollution episodes were observed, including a daily concentration of 552 µg/m³ on 12 January 2013.

Reliable next-day PM2.5 forecasts can support public-health alerts, decisions concerning school closures, temporary restrictions on traffic and industrial activity, and individual exposure-reduction measures. The forecasting problem is also relevant to Thailand, particularly during the cool, dry season in Bangkok and the biomass-burning season in Chiang Mai.

This project evaluates machine-learning approaches to daily PM2.5 prediction and examines how meteorological conditions, antecedent pollution, and temporal indicators contribute to model predictions.

## 2. Dataset Source

| Attribute | Description |
| --- | --- |
| Dataset | Beijing PM2.5 Data |
| Source and licence | Chen, S. (2015). Beijing PM2.5 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5JS49.. Licensed under CC BY 4.0. |
| Temporal coverage | 43,824 hourly records spanning 1 January 2010 to 31 December 2014. |
| Measurements | PM2.5 concentrations measured at the United States Embassy in Beijing; meteorological measurements from Beijing Capital International Airport, including dew point, temperature, atmospheric pressure, combined wind direction, cumulative wind speed, and accumulated hours of rainfall and snowfall. |

## 3. Research Questions

1. Prediction: how accurately can ML forecast tomorrow’s PM2.5, and which model is best?

2. Drivers: which factors drive the forecast, and in which direction?

3. Information: how much comes from the pollution already in the air, and how much from the weather?

4. Application: can it warn of polluted days, and did the APEC 2014 emission controls work?

## 4. Methodology

### 4.1. Data Quality Assessment and Preprocessing

The hourly time series was assessed for temporal continuity, duplicate records, missing observations, and outliers. No temporal gaps or duplicate records were identified. Missing values occurred only in PM2.5 measurements and represented 4.7% of hourly records. The 1.5 × interquartile range criterion identified 4.2% of hourly observations as potential outliers. These observations were retained because extreme values were associated with documented environmental events, including Lunar New Year fireworks, the January 2013 haze episode, and a March 2010 dust storm.

Hourly wind speed was reconstructed from the cumulative wind-speed variable, Iws. Relative humidity was estimated from temperature and dew point using the Magnus formula. Hourly observations were aggregated to daily values, and a daily PM2.5 mean was accepted only when at least 20 valid hourly measurements were available, following the GB 3095-2012 criterion. This procedure yielded 1,711 valid daily observations.

### 4.2. Feature Engineering and Prediction Setting

The predictor set comprised 21 features:

- Thirteen meteorological features: temperature, daily temperature range, dew point, relative humidity, atmospheric pressure, wind speed, wind-direction proportions, rainfall and snowfall hours, and day-to-day changes in pressure and dew point.

- Three persistence features: the previous day's daily mean PM2.5, the previous evening's PM2.5 concentration during 18:00–23:00, and the daily mean concentration two days earlier.

- Five calendar features: cyclic representations of day of year and indicators for weekends, the heating season, and Spring Festival.

The target was the mean PM2.5 concentration on the forecast day. Observed meteorological conditions for that day were used as a proxy for a perfect weather forecast, constituting a perfect-prognosis experiment. The resulting performance therefore reflects prediction under idealised meteorological information rather than a fully operational forecast setting.

### 4.3. Unsupervised Analysis of Meteorological Regimes

K-means clustering was fitted exclusively to the training years, and the resulting weather regimes were visualized using principal component analysis (PCA). Nine clustering variables were selected using domain knowledge, variance assessment, and correlation analysis without reference to the prediction target. A sensitivity analysis involving six alternative variable sets consistently identified a relatively clean, windy regime with a mean PM2.5 concentration of approximately 47 µg/m³ and a calm winter regime with concentrations approximately 2.7 times higher.

### 4.4. Chronological Partitioning and Candidate Models

A chronological partition was used to reduce temporal leakage. The modelling sample contained 1,276 training days from 2010–2013 and 342 test days from 2014.

Ten models were evaluated: multiple linear regression, Elastic Net, k-nearest neighbours (KNN), support vector regression with a radial basis function kernel (RBF-SVR), a decision tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, and a stacking ensemble. The stacking model combined Random Forest, XGBoost, LightGBM, and RBF-SVR through a linear meta-learner with non-negative coefficients. Scaling was implemented within Pipeline objects for the linear, distance-based, and kernel-based models.

### 4.5. Hyperparameter Optimization and Model Selection

Hyperparameters were optimized using annual forward-chaining cross-validation, with root mean squared error (RMSE) as the selection criterion:

1. Training on 2010 and validation on 2011.

2. Training on 2010–2011 and validation on 2012.

3. Training on 2010–2012 and validation on 2013.

GridSearchCV was applied to five models with relatively small search spaces of 44–105 combinations each. For Gradient Boosting, XGBoost, and LightGBM, RandomizedSearchCV evaluated 150 configurations per model from search spaces containing approximately 0.16–1.3 million combinations. Under the random-search sampling assumptions, 150 draws provide a probability greater than 99.9% of sampling at least one configuration within the best-performing 5% of the search space.

An automated boundary assessment examined all 33 tuned parameter values. Each selected value was located within its search range, at a natural limit, or at a boundary for which extending the range did not improve cross-validation RMSE. Validation errors were also compared on a paired, year-by-year basis. The final model was selected using cross-validation RMSE; the 2014 test set was reserved for the final evaluation.

### 4.6. Predictive Evaluation and Information-Value Analysis

Test performance was assessed using the coefficient of determination (R²), mean absolute error (MAE), and RMSE. Residuals were examined across pollution levels. Health-alert performance was evaluated for observed concentrations exceeding 75 µg/m³ using a confusion matrix, precision, recall, F1 score, and a decision-threshold sweep. A higher-recall alert threshold was selected using validation-year predictions rather than test-year outcomes.

To assess the contribution of different information sources, the stacking model was retrained using alternative predictor groups: persistence-based information, forecast-day meteorological information, previous-day meteorological information, and the complete feature set. Calendar predictors were included in the reduced-feature comparisons reported below.

### 4.7. Model Interpretation and Environmental Case Study

Interpretation combined built-in feature importance from Random Forest and XGBoost, permutation importance, and SHapley Additive exPlanations (SHAP) for the stacking ensemble. Model-agnostic shap.PermutationExplainer was used with all training days as the background dataset. Reliability was assessed by repeating the analysis with alternative random feature orderings and comparing the rankings with exact TreeSHAP explanations for XGBoost and Random Forest. SHAP dependence plots, partial dependence plots, and individual-day waterfall plots were also examined.

A weather-only stacking model was used for meteorological normalisation of PM2.5 concentrations during the APEC 2014 emission-control period.

## 5. Main Results

### 5.1. Model Comparison

Cross-validation results refer to 2011–2013; all test metrics refer to the independent 2014 test set of 342 days. Complete scores are provided in [results/model_comparison.csv](results/model_comparison.csv), and selected hyperparameters are documented in [results/best_hyperparameters.json](results/best_hyperparameters.json).

| Model | CV RMSE (2011–2013; µg/m³) | Test R² | Test MAE (µg/m³) | Test RMSE (µg/m³) |
| --- | --- | --- | --- | --- |
| Stacking (RF + XGBoost + LightGBM + SVR) | 33.6 | 0.855 | 22.7 | 31.3 |
| SVR (RBF) | 34.1 | 0.828 | 24.5 | 34.1 |
| Gradient Boosting | 35.7 | 0.843 | 23.5 | 32.5 |
| XGBoost | 36.0 | 0.841 | 23.8 | 32.8 |
| LightGBM | 36.1 | 0.837 | 24.2 | 33.1 |
| Random Forest | 37.6 | 0.837 | 24.4 | 33.1 |
| Multiple Linear Regression | 40.3 | 0.770 | 29.4 | 39.4 |
| Elastic Net | 40.3 | 0.770 | 29.4 | 39.4 |
| KNN | 43.1 | 0.767 | 28.6 | 39.6 |
| Decision Tree | 45.7 | 0.731 | 30.2 | 42.6 |
| *Persistence baseline (tomorrow = today)* | – | 0.207 | 51.6 | 73.1 |

Comparison of cross-validation and test performance across candidate models

### 5.2. Health-Alert Performance

For identifying days with observed PM2.5 concentrations above 75 µg/m³, the selected model achieved 91% recall and 84% precision, with an F1 score of 0.87, compared with 0.70 for the persistence baseline. Reducing the prediction threshold to 70 µg/m³, as selected using the 2011–2013 validation predictions, increased recall to 93% while reducing precision to 80%. This illustrates the trade-off between detecting polluted days and issuing false alarms.

### 5.3. Contribution of Predictor Groups and Extreme-Event Performance

The persistence-plus-calendar model achieved R² = 0.52, whereas the meteorology-plus-calendar model achieved R² = 0.61. Combining all predictor groups increased R² to 0.855. When previous-day weather was used in place of forecast-day weather, R² decreased to 0.54. These comparisons indicate substantial predictive value from meteorological information for the forecast day.

Despite its overall performance, the selected model underestimated concentrations on the most polluted days, defined as those exceeding 250 µg/m³, by approximately 46 µg/m³ on average.

## 6. Selected Stacking Ensemble for the Best Model

The selected ensemble combined the tuned Random Forest, XGBoost, LightGBM, and RBF-SVR models using a linear meta-learner constrained to non-negative coefficients.

The ensemble achieved the lowest cross-validation RMSE among the ten candidate models, at 33.6 µg/m³. It outperformed RBF-SVR, the strongest individual model according to cross-validation, in two of the three validation years. The overall cross-validation improvement was modest, at 0.55 µg/m³.

On the independent 2014 test set, the ensemble achieved R² = 0.855, MAE = 22.7 µg/m³, and RMSE = 31.3 µg/m³. Its RMSE was approximately 57% lower than that of the persistence baseline. The difference between training and test R² was 0.03.

The fitted meta-learner coefficients were 0.53 for RBF-SVR, 0.44 for LightGBM, 0.08 for Random Forest, and 0 for XGBoost. These are non-negative regression coefficients rather than necessarily normalised proportions. The zero coefficient for XGBoost indicates no additional contribution within this fitted combination; it does not imply that XGBoost lacked predictive value as an individual model. The fitted ensemble suggests complementarity between the kernel-based and tree-based approaches.

## 7. Feature Attribution and Model Explainability

SHAP values were computed directly for the stacking ensemble using the model-agnostic PermutationExplainer, because TreeSHAP is not directly applicable to an ensemble containing an SVR component. The reference value was the mean prediction over the training days, 98.7 µg/m³. The SHAP attributions sum to the difference between this reference value and each prediction.

SHAP summary for the stacking ensemble

| Rank | SHAP of the stacking model, mean SHAP (µg/m³) | Permutation importance of the stacking model, RMSE increase (µg/m³) |
| --- | --- | --- |
| 1 | PM_eve_lag1: yesterday 18–23 h PM2.5 (29.68) | PM_eve_lag1 (+37.16) |
| 2 | dDEWP: change in dew point (15.04) | dDEWP (+12.65) |
| 3 | RH: relative humidity (15.04) | RH (+9.01) |
| 4 | DEWP: dew point (11.02) | WS (+7.55) |
| 5 | WS: wind speed (10.37) | PM_lag1 (+7.18) |
| 6 | TEMP: temperature (9.88) | DEWP (+6.07) |

Repeating the SHAP calculation with alternative random orderings changed the attribution values by an average of 0.09 µg/m³. The SHAP and permutation-importance rankings shared four of their five highest-ranked predictors. The exact TreeSHAP rankings for XGBoost and Random Forest shared three of the top five predictors with the stacking-model SHAP ranking. Previous-evening PM2.5 ranked first in all four rankings, while the change in dew point and wind speed appeared among the top five in each.

Grouped SHAP attribution assigned 63% of the total attribution to meteorological predictors, 28% to persistence predictors, and 9% to calendar predictors. These percentages describe attribution within the fitted model and should not be interpreted as causal contributions to ambient pollution.

## 8. Environmental Interpretation, Implications, and Limitations

### 8.1. Persistence and Synoptic Meteorology

Elevated previous-evening PM2.5 was associated with higher next-day predictions, consistent with the accumulation of pollution during persistent stagnant conditions. An evening concentration of 470 µg/m³ was associated with a positive model contribution of approximately 131 µg/m³.

Sharp decreases in dew point of at least 5.8 °C were associated with forecast reductions of approximately 32 µg/m³, while pressure increases of at least 5 hPa were associated with reductions of approximately 12 µg/m³. These patterns are consistent with the passage of dry north-westerly air from the Mongolian Plateau. Conversely, increasing dew point and decreasing pressure were associated with positive contributions of approximately 28 and 14 µg/m³, respectively, consistent with moist southerly transport from the industrial North China Plain.

### 8.2. Humidity, Temperature, and Wind Speed

Relative humidity was associated with positive model contributions of approximately 27 µg/m³ on the most humid 10% of days and negative contributions of approximately 31 µg/m³ on the driest days. This pattern is consistent with humidity-related secondary aerosol formation and particle growth, although the attribution analysis does not establish these mechanisms causally. Dew point, temperature, and relative humidity are strongly interdependent, so their individual SHAP values should be interpreted jointly.

The calmest 10% of days, with wind speeds of 1.4 m/s or less, were associated with positive contributions of approximately 34 µg/m³. Above approximately 3.5 m/s, the model indicated diminishing additional reductions with increasing wind speed. The windiest 10% of days were associated with negative contributions of approximately 14 µg/m³.

### 8.3. Calendar Effects and Seasonal Confounding

No measurable weekend effect was identified. The Spring Festival indicator contributed approximately 12 µg/m³ on festival days, but this relationship was estimated from only eight training observations and should therefore be interpreted cautiously.

Higher winter concentrations were represented by meteorological and seasonal predictors that were strongly correlated with the heating-season indicator. Consequently, these data and model explanations could not distinguish the independent contribution of winter heating from that of seasonal meteorology.

### 8.4. APEC 2014 Meteorological Normalization

During the emission-control period of 3–12 November 2014, observed PM2.5 concentrations were 60% lower than in the surrounding weeks.

The weather-only model estimated a 32% reduction attributable to meteorological differences alone. Observed concentrations during the control period were nevertheless 37% below the weather-expected level. Under the study's meteorological normalization procedure, concentrations were estimated to be 41% lower than the business-as-usual reference after adjustment for weather. These percentages refer to distinct comparisons and are not directly additive. Taken together, the results suggest substantial contributions from both favorable meteorology and emission controls, with an approximately comparable contribution from each in the study's interpretation.

The case study covers only one 10-day intervention period. A discrepancy of comparable magnitude occurred in 6.4% of the 10-day windows examined in 2014. The evidence is therefore suggestive rather than conclusive and does not establish a definitive causal effect of the intervention.

Meteorological normalization of PM2.5 during the APEC 2014 emission-control period

### 8.5. Implications for Environmental Management

The results support integrating air-quality monitoring with meteorological forecasts in early-warning systems. Where the consequences of missed pollution events exceed those of false alarms, a higher-recall decision threshold may be appropriate, subject to the needs of the responsible public-health authority.

The APEC analysis is consistent with a beneficial contribution from short-term emission controls, while demonstrating the importance of accounting for meteorological variability when assessing interventions. Machine-learning-based meteorological normalization offers a practical analytical approach, although its findings require cautious interpretation.

The workflow could be adapted to Bangkok and Chiang Mai using Pollution Control Department data from Air4Thai and satellite-derived fire counts. Such an application would require local training and validation rather than assuming direct transferability of the Beijing model.
