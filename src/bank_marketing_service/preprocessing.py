import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def prepare_features(df):
    df_copy = df.copy()

    df_copy = df_copy.replace("unknown", np.nan)

    df_copy["pdays"] = np.where(df_copy["pdays"] != 999, 1, 0)

    return df_copy


def get_preprocessor_scaled():

    numeric_cols = ["age", "campaign", "pdays", "previous",
                    "emp.var.rate", "cons.price.idx", "cons.conf.idx",
                    "euribor3m", "nr.employed"]
    onehot_cols = ["job", "marital", "poutcome", "day_of_week", "month"]
    ord_col = ["education"]
    binary_cols = ["default", "housing", "loan", "contact"]

    ohe_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    ord_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(categories=[
            ["illiterate", "basic.4y", "basic.6y", "basic.9y",
             "high.school", "professional.course", "university.degree"]
        ])),
    ])

    bin_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ord", OrdinalEncoder(categories=[
            ["no", "yes"],              # default
            ["no", "yes"],              # housing
            ["no", "yes"],              # loan
            ["cellular", "telephone"],  # contact
        ])),
    ])

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("ohe", ohe_pipe, onehot_cols),
        ("ord", ord_pipe, ord_col),
        ("bin", bin_pipe, binary_cols),
    ])

    return preprocessor