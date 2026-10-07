import os

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)


# Load the census.csv data.
project_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_path, "data", "census.csv")
print(data_path)

data = pd.read_csv(data_path, skipinitialspace=True)
data.columns = data.columns.str.strip()

# Remove surrounding spaces from text values.
for column in data.select_dtypes(include=["object"]).columns:
    data[column] = data[column].str.strip()

# Split the provided data into train and test datasets.
train, test = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
    stratify=data["salary"],
)

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Process the training data and fit the encoders.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

# Process the test data using the fitted encoders.
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

# Train the model on the training dataset.
model = train_model(X_train, y_train)

# Create the model folder.
model_directory = os.path.join(project_path, "model")
os.makedirs(model_directory, exist_ok=True)

# Save the model and the encoders.
model_path = os.path.join(model_directory, "model.pkl")
encoder_path = os.path.join(model_directory, "encoder.pkl")
lb_path = os.path.join(model_directory, "lb.pkl")

save_model(model, model_path)
save_model(encoder, encoder_path)
save_model(lb, lb_path)

# Load the model.
model = load_model(model_path)

# Run model inference on the test dataset.
preds = inference(model, X_test)

# Calculate and print the metrics.
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# Compute performance on categorical slices.
slice_output_path = os.path.join(project_path, "slice_output.txt")

with open(slice_output_path, "w") as f:
    for col in cat_features:
        for slicevalue in sorted(test[col].unique()):
            count = test[test[col] == slicevalue].shape[0]

            p, r, fb = performance_on_categorical_slice(
                test,
                col,
                slicevalue,
                cat_features,
                "salary",
                encoder,
                lb,
                model,
            )

            slice_heading = f"{col}: {slicevalue}, Count: {count:,}"
            slice_metrics = (
                f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}"
            )

            print(slice_heading)
            print(slice_metrics)
            print(slice_heading, file=f)
            print(slice_metrics, file=f)
            print(file=f)
