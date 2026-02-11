# Model Comparison with Cross-Validation
# Random Forest vs XGBoost vs CatBoost

import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

filepath = r"C:\Users\felix\ames_housing-ml\data\AmesHousing.csv"
df = pd.read_csv(filepath)

X = df.drop(columns=["SalePrice"])
y = np.log1p(df["SalePrice"])

X = X.drop(columns=["Order", "PID"], errors="ignore")

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numerical_cols = X.select_dtypes(exclude="object").columns.tolist()
cat_features = [X.columns.get_loc(col) for col in categorical_cols]


preprocess = ColumnTransformer(
    transformers=[
        (
            "cat",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
            ),
            categorical_cols,
        ),
        ("num", "passthrough", numerical_cols),
    ]
)

models = {
    "CatBoost": CatBoostRegressor(
        iterations=1000,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0,
    ),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = []

for name, model in models.items():
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", model),
        ]
    )

    scores = cross_val_score(
        pipe,
        X,
        y,
        scoring="neg_mean_absolute_error",
        cv=kf,
        n_jobs=-1,
    )

    mae_mean = -scores.mean()
    mae_std = scores.std()

    results.append({
        "model": name,
        "mae_mean": mae_mean,
        "mae_std": mae_std,
    })

results_df = pd.DataFrame(results).sort_values("mae_mean")
print(results_df)
