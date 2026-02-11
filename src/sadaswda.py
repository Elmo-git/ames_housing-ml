import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

filepath = r"C:\Users\felix\ames_housing-ml\data\AmesHousing.csv"
df = pd.read_csv(filepath)


df = df.drop(columns=["Order", "PID"], errors="ignore")

X = df.drop(columns=["SalePrice"])
categorical_cols = X.select_dtypes(include="object").columns.tolist()


X_train, X_valid, y_train, y_valid = train_test_split(
    X, df["SalePrice"], test_size=0.2, random_state=42
)

model_normal = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    verbose=0,
    random_seed=42
)

model_normal.fit(X_train, y_train, cat_features=categorical_cols)
pred_normal = model_normal.predict(X_valid)

residual_normal = y_valid - pred_normal


y_train_log = np.log1p(y_train)

model_log = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    verbose=0,
    random_seed=42
)

model_log.fit(X_train, y_train_log, cat_features=categorical_cols)

pred_log = np.expm1(model_log.predict(X_valid))
residual_log = y_valid - pred_log



plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.hist(residual_normal, bins=50)
plt.title("Residual - Normal Target")

plt.subplot(1,2,2)
plt.hist(residual_log, bins=50)
plt.title("Residual - Log Target")

plt.show()
