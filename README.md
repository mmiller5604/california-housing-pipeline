# California Housing Modeling Pipeline

This contains a full end-to-end pipeline for predicting California house prices, featuring:

- Data cleaning & feature engineering (ratios, polynomial terms, geo-clusters)
- Bayesian hyperparameter tuning of XGBoost
- Explainability via SHAP
- Feature pruning & model stacking (RF + XGB + Lasso)
- Spatial residual visualization with Folium


## Quickstart


pip install -r requirements.txt
python california_housing_prediction.py
