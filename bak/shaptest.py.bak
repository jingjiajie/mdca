import numpy as np
import pandas as pd
import xgboost
import shap

# train an XGBoost model
X, y = shap.datasets.adult()
print(X)
y = pd.Series(y.tolist())
y = y.map({True: 1, False: 0})
print(y)
model = xgboost.XGBRegressor().fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)

# visualize the first prediction's explanation
# shap.plots.waterfall(shap_values[0])
shap.plots.beeswarm(shap_values)