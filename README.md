California Housing Price Pipeline
I set out to build an end-to-end pipeline that turns raw California housing data into spot-on price predictions—and came away with some pretty cool takeaways:

Smart Features

I cooked up ratios like rooms-per-household and bedrooms-per-room, plus squared and interaction terms on income.

I even clustered lat/long into “neighborhood” groups with K-means and binned incomes into quintiles.

Tuned XGBoost

Wrapping the model in a log-transform helped stabilize things.

I ran a Bayesian search to hone in on the best n_estimators, max_depth, learning rate, regularization, etc.

Result: CV RMSE ≈ 45 860 and R² ≈ 0.843—already a leap over my vanilla Random Forest.

Reading the “Why” with SHAP

I generated a SHAP summary to see exactly which features are driving predictions.

Top hits: median income (no surprise), inland vs. coastal, then geography (lat/long).

My custom ratios show up solidly in the middle—proof that the extra crafting paid off.

Pruning & Stacking

I dropped the bottom 5 % of features by importance (traded off a little accuracy, RMSE ≈ 47 200).

Then I built a stacked ensemble (XGB + RF + Lasso) that squeezed RMSE down to ≈ 44 350 on the test set.

Spotting Trouble Zones

Mapping residuals back on a Folium map showed me where the model over- or under-predicts—great clues for adding local amenities or school-district features next.

Bottom line: from raw CSV to a 44 k-dollar RMSE, this pipeline has taught me tons about feature engineering, hyperparameter search, explainability, and ensembling. Feel free to poke around the code, browse the SHAP plot, or click the residuals_map.html to explore where we’re still missing the mark.

## Quickstart


pip install -r requirements.txt
python california_housing_prediction.py
