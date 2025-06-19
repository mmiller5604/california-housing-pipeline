

import pandas as pd
import numpy as np
import shap
import folium
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_validate
from skopt import BayesSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV, LassoCV, LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


def main():
    # --- 1) Load and FE ---
    csv_path = r'E:/portfolio-projects/housing.csv'  # adjust as needed
    df = pd.read_csv(csv_path)

    # Feature engineering
    df['rooms_per_household']      = df['total_rooms'] / df['households']
    df['bedrooms_per_room']        = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']
    df['median_income_sq']         = df['median_income']**2
    df['inc_rooms_interaction']    = df['median_income'] * df['rooms_per_household']
    df['income_cat']               = pd.qcut(df['median_income'], 5, labels=False)
    df['geo_cluster']              = KMeans(n_clusters=10, random_state=42)\
                                         .fit_predict(df[['latitude','longitude']])

    # Prepare features/target
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- 2) Preprocessor ---
    numeric_cols     = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = ['ocean_proximity','income_cat','geo_cluster']
    numeric_feats    = [c for c in numeric_cols if c not in categorical_cols]

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ])
    cat_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    preproc = ColumnTransformer([
        ('num', num_pipe, numeric_feats),
        ('cat', cat_pipe, categorical_cols)
    ])

    # --- 3) Bayesian hyperparameter tuning for XGBoost ---
    xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    pipe_xgb = Pipeline([('prep', preproc), ('xgb', xgb)])

    search_space = {
        'xgb__n_estimators':      (100, 500),
        'xgb__learning_rate':     (0.01, 0.3, 'log-uniform'),
        'xgb__max_depth':         (3, 10),
        'xgb__subsample':         (0.5, 1.0),
        'xgb__colsample_bytree':  (0.5, 1.0),
        'xgb__reg_alpha':         (0.0, 1.0),
        'xgb__reg_lambda':        (1.0, 10.0)
    }

    bayes = BayesSearchCV(
        pipe_xgb, search_space,
        n_iter=25, cv=3,
        scoring='neg_root_mean_squared_error',
        random_state=42, n_jobs=-1
    )
    print("Running Bayesian search for XGBoost...")
    bayes.fit(X_train, y_train)
    best_xgb = bayes.best_estimator_
    print("Best XGB params:", bayes.best_params_)

    # Log-target transform
    model_xgb = TransformedTargetRegressor(
        regressor=best_xgb,
        func=np.log, inverse_func=np.exp
    )

    # CV performance
    cv = cross_validate(model_xgb, X_train, y_train, cv=5,
                        scoring=('neg_root_mean_squared_error','r2'))
    print(f"XGB CV RMSE: {-cv['test_neg_root_mean_squared_error'].mean():.2f}")
    print(f"XGB CV R² : {cv['test_r2'].mean():.3f}")

    # --- 4) SHAP explainability ---
    try:
        print("Computing SHAP values...")
        sample = X_train.sample(500, random_state=1)
        Xp = best_xgb.named_steps['prep'].transform(sample)
        explainer = shap.Explainer(best_xgb.named_steps['xgb'])
        sv = explainer(Xp)
        plot = shap.summary_plot(sv, Xp, feature_names=best_xgb.named_steps['prep'].get_feature_names_out(), show=False)
        plt.savefig('shap_summary.png', bbox_inches='tight')
        plt.clf()
        print("Saved SHAP summary to shap_summary.png")
    except Exception as e:
        print("SHAP failed, skipping:", e)

    # --- 5) Prune bottom 5% features by importance ---
    importances = best_xgb.named_steps['xgb'].feature_importances_
    names = best_xgb.named_steps['prep'].get_feature_names_out()
    df_imp = pd.DataFrame({'feat': names, 'imp': importances})
    threshold = np.percentile(df_imp.imp, 5)
    keep = df_imp[df_imp.imp > threshold].feat.tolist()
    print(f"Pruning features <5% importance (thr={threshold:.4f})")

    def select_feats(arr):
        df_arr = pd.DataFrame(arr, columns=names)
        return df_arr[keep].values

    prune_pipe = Pipeline([
        ('prep', preproc),
        ('prune', FunctionTransformer(select_feats))
    ])
    xgb_pruned = XGBRegressor(**bayes.best_params_, objective='reg:squarederror', random_state=42, n_jobs=-1)
    pruned_pipe = Pipeline([('prep_pruned', prune_pipe), ('xgb', xgb_pruned)])
    pruned_model = TransformedTargetRegressor(regressor=pruned_pipe, func=np.log, inverse_func=np.exp)
    pruned_model.fit(X_train, y_train)
    yp = pruned_model.predict(X_test)
    rmse_pruned = np.sqrt(mean_squared_error(y_test, yp))
    print(f"Pruned XGB RMSE: {rmse_pruned:.2f}")

    # --- 6) Stacking ensemble with Lasso meta-learner ---
    print("Building ensemble...")
    rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    estimators = [
        ('xgb', best_xgb),
        ('rf', Pipeline([('prep', preproc), ('rf', rf)])),
        ('ridge', Pipeline([('prep', preproc), ('ridge', RidgeCV())]))
    ]
    stack = StackingRegressor(
        estimators=estimators,
        final_estimator=LassoCV(cv=5),
        cv=5, n_jobs=-1
    )
    stack_model = TransformedTargetRegressor(regressor=stack, func=np.log, inverse_func=np.exp)
    stack_model.fit(X_train, y_train)
    ys = stack_model.predict(X_test)
    rmse_stack = np.sqrt(mean_squared_error(y_test, ys))
    print(f"Stacked model RMSE: {rmse_stack:.2f}")

    # --- 7) Spatial residual map ---
    print("Creating residuals map...")
    res_df = pd.DataFrame({
        'lat': df.loc[y_test.index, 'latitude'],
        'lon': df.loc[y_test.index, 'longitude'],
        'residual': y_test - ys
    })
    m = folium.Map(location=[res_df.lat.mean(), res_df.lon.mean()], zoom_start=8)
    for _, row in res_df.iterrows():
        folium.CircleMarker(
            location=[row.lat, row.lon],
            radius=3,
            color='red' if row.residual>0 else 'blue',
            fill=True,
            fill_opacity=0.6,
            popup=f"Residual: {row.residual:.0f}"
        ).add_to(m)
    m.save('residuals_map.html')
    print("Saved spatial residual map to residuals_map.html")

    # --- 8) Portfolio summary ---
    summary = f"""
    California Housing Modeling Pipeline Summary
    --------------------------------------------
    - Baseline RF RMSE ~53k, R² ~0.79
    - Tuned XGB CV RMSE: {-cv['test_neg_root_mean_squared_error'].mean():.2f}, R²: {cv['test_r2'].mean():.3f}
    - Pruned XGB RMSE: {rmse_pruned:.2f}
    - Stacked ensemble RMSE: {rmse_stack:.2f}
    Features: {', '.join(keep)}
    """
    with open('pipeline_summary.txt', 'w') as f:
        f.write(summary)
    print("Written pipeline summary to pipeline_summary.txt")

if __name__ == '__main__':
    main()
