import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

from ml.data import process_data


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.

    Returns
    -------
    model
        Trained machine learning model.
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.base.ClassifierMixin
        Trained machine learning model.
    X : np.array
        Data used for prediction.

    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def save_model(model, path):
    """
    Serializes model to a file.

    Inputs
    ------
    model
        Trained model, OneHotEncoder, or LabelBinarizer.
    path : str
        Path to save pickle file.
    """
    with open(path, "wb") as file:
        pickle.dump(model, file)


def load_model(path):
    """Loads pickle file from `path` and returns it."""
    with open(path, "rb") as file:
        return pickle.load(file)


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features,
    label, encoder, lb, model
):
    """
    Computes model metrics on a specified categorical slice.

    Processes the slice using the trained categorical encoder
    and label binarizer.

    Inputs
    ------
    data : pd.DataFrame
        Dataframe containing the features and label.
    column_name : str
        Column containing the sliced feature.
    slice_value : str, int, float
        Value of the slice feature.
    categorical_features : list
        Names of the categorical features.
    label : str
        Name of the label column.
    encoder : sklearn.preprocessing.OneHotEncoder
        Trained categorical encoder.
    lb : sklearn.preprocessing.LabelBinarizer
        Trained label binarizer.
    model : sklearn.base.ClassifierMixin
        Trained machine learning model.

    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    slice_data = data.loc[data[column_name] == slice_value]

    if slice_data.empty:
        raise ValueError(
            f"No rows found for {column_name}={slice_value!r}."
        )

    X_slice, y_slice, _, _ = process_data(
        slice_data,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta
