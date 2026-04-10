from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def _to_float_df(X):
    """
    Convert all columns to numeric and then float.
    This avoids mixed dtypes (especially bool/int/float mixes) becoming object
    during LIME perturbation -> pandas reconstruction -> model.predict().
    """
    X_num = X.copy()

    # Convert bools first so everything can become float cleanly
    bool_cols = X_num.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        X_num[bool_cols] = X_num[bool_cols].astype(int)

    # Coerce everything numeric, then cast to float
    X_num = X_num.apply(pd.to_numeric, errors='coerce').fillna(0).astype(float)
    return X_num


def make_lime_explainer(X_train):
    X_train_num = _to_float_df(X_train)

    explainer = LimeTabularExplainer(
        training_data=X_train_num.to_numpy(dtype=float),
        feature_names=X_train_num.columns.tolist(),
        mode='regression',
        discretize_continuous=True,
        random_state=42
    )
    return explainer


def explain_single_instance_lime(model, explainer, X_data, y_data, row_idx=0, num_features=12):
    X_data_num = _to_float_df(X_data)

    instance = X_data_num.iloc[row_idx]
    data_index = X_data_num.index[row_idx]

    # Keep one-row prediction input as float DataFrame with original columns
    instance_df = pd.DataFrame([instance.values], columns=X_data_num.columns).astype(float)

    pred_value = model.predict(instance_df)[0]
    true_value = y_data.loc[data_index]

    def predict_fn(x):
        # x comes from LIME as ndarray; rebuild with original columns and force float
        x_df = pd.DataFrame(x, columns=X_data_num.columns)
        x_df = _to_float_df(x_df)
        return model.predict(x_df)

    exp = explainer.explain_instance(
        data_row=instance.to_numpy(dtype=float),
        predict_fn=predict_fn,
        num_features=num_features
    )

    return exp, instance, pred_value, true_value, data_index


def lime_explanation_to_df(exp):
    lime_df = pd.DataFrame(exp.as_list(), columns=['feature', 'weight'])
    lime_df['abs_weight'] = lime_df['weight'].abs()
    lime_df = lime_df.sort_values('abs_weight', ascending=False).reset_index(drop=True)
    return lime_df


def plot_lime_explanation(exp, title=None):
    fig = exp.as_pyplot_figure()
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()