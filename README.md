# MARS-charts
Implementation of MARS charts and MARS metrics for evaluating classifier exclusivity: the comparative uniqueness of binary classifier predictions.

## Requirements
Python 3.7.9 <br>
Pandas 1.1.3 <br>
Plotly 5.4.0

## Dataset
Performed classififcation on Breast cancer dataset from scikit learn using Logistic Regression, Decision Tree, Random Forest and Support Vcetor Machine. Classification results are arranged in the proposed format same as classification_breast_cancer_dataset.csv.

MARS Charts and MARS Metrics User Instructions:
To produce the results:

Upload a CSV (comma-separated-value) text file with the classification decisions for the classifiers you are comparing, in the following format:

1.	Column 1: OBSERVATION_ID: The observation ID’s for the observation that was classified by the classifier.  Duplicate values permitted in this column, since multiple classifiers will make classification decisions for each observationID.
2.	Column 2: CLASSIFIER_NAME: Name of the classifier. Duplicate values permitted, as the classifier was applied (by you) to each observationID
3.	Column 3: PREDICTED_CLASS: The class predicted by the respective classifier for respective observation ID.  This should be a 1 or 0 value, where 1 indicates that classifier predicted the observation was in the target class, and 0 indicates the classifier predicted the observation was not in the target class.
4.	Column 4: ACTUAL_CLASS: Actual (true) class respective to that observation and classifier.  This should be a 1 or 0 value, where 1 indicates the observation was in the target class, and 0 indicates the observation was not in the target class.


## Evaluation

How to interpret the results:
Refer to the document documentation.txt

## License
Distributed under the MIT License.

## Contact
mars_classifier_evaluation@vt.edu 
