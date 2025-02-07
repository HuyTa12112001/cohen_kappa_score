import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("train.csv")

X = df["id_code"]
y = df["diagnosis"]

X_train, X_test, y_train, y_test = train_test_split(
  X,y , random_state=104, test_size=0.2, shuffle=True)

df["id_code"] = X_test
df["diagnosis"] = y_test
df.to_csv("train-2.csv", index=False)

df.dropna().to_csv("train-2.csv")
