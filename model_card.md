# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This project uses a Random Forest classifier with 100 trees and a
random seed of 42. OneHotEncoder processes categorical features,
and LabelBinarizer encodes salary labels. The saved model provides
predictions through a FastAPI application.

## Intended Use
The model predicts whether annual income is <=50K or >50K. It is
intended for educational purposes and should not be used for
employment, lending, or other decisions affecting individuals.

## Training Data
The model uses 80% of the provided census.csv dataset for training.
The split is stratified by salary with a random seed of 42.
Surrounding spaces are removed, categorical features are encoded,
and numeric features remain unscaled. Encoders are fitted only
on training data. Question-mark values remain as categories.

## Evaluation Data
The remaining 20%, containing 6,513 records, is used for evaluation
with the training encoders. Performance is also measured across
all eight categorical features and saved in slice_output.txt.

## Metrics
Precision, recall, and F1 measure performance for the >50K class.
The model achieved precision of 0.7353, recall of 0.6378, and an
F1 score of 0.6831 on the test set.

Slice results vary. For example, Doctorate has an F1 of 0.8906,
while HS-grad has an F1 of 0.4861. Undefined metrics receive 1.0
because the code uses zero_division=1; these values do not
necessarily indicate perfect performance.

## Ethical Considerations
Features such as race and sex may reflect historical inequalities.
The model may reproduce these biases. Slice metrics alone do not
establish fairness, and further analysis is needed before real-world use.

## Caveats and Recommendations
Results are based on one train-test split, and small slices may
have unreliable scores. Performance on external data and unseen
categories has not been established. Future improvements include
cross-validation, hyperparameter tuning, baseline comparisons,
and additional fairness evaluation.
