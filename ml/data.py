import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X, categorical_features=None, label=None,
    training=True, encoder=None, lb=None
):
    """
    Process the data used in the machine learning pipeline.

    Processes categorical features using one hot encoding and
    labels using a label binarizer. Supports training and inference.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and optional label.
    categorical_features : list[str]
        Names of the categorical features.
    label : str
        Label column name. If None, returns an empty label array.
    training : bool
        Indicator for training or inference/validation mode.
    encoder : sklearn.preprocessing.OneHotEncoder
        Trained encoder, required when training=False.
    lb : sklearn.preprocessing.LabelBinarizer
        Trained label binarizer for labeled validation data.

    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels, or an empty array for unlabeled data.
    encoder : sklearn.preprocessing.OneHotEncoder
        Fitted categorical encoder.
    lb : sklearn.preprocessing.LabelBinarizer
        Fitted label binarizer.
    """
    if categorical_features is None:
        categorical_features = []

    if label is not None:
        y = X[label]
        X = X.drop(columns=[label])
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(
        columns=categorical_features
    ).to_numpy()

    if training:
        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
        )
        X_categorical = encoder.fit_transform(X_categorical)

        if label is not None:
            lb = LabelBinarizer()
            y = lb.fit_transform(y.to_numpy()).ravel()
    else:
        if encoder is None:
            raise ValueError(
                "A fitted encoder is required when training=False."
            )

        X_categorical = encoder.transform(X_categorical)

        if label is not None:
            if lb is None:
                raise ValueError(
                    "A fitted label binarizer is required."
                )
            y = lb.transform(y.to_numpy()).ravel()

    X = np.concatenate(
        [X_continuous, X_categorical],
        axis=1,
    )
    return X, y, encoder, lb


def apply_label(inference):
    """
    Convert a binary prediction into a salary label.

    Assumes the model uses 0 for <=50K and 1 for >50K.
    """
    if inference[0] == 1:
        return ">50K"
    elif inference[0] == 0:
        return "<=50K"

    raise ValueError("The prediction must be either 0 or 1.")
