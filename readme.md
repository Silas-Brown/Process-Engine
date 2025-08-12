## Overview 
Process Engine is an automated preprocessor that lets you define all of your preprocessing logic in a highly readable format and executes it leakage-free, all in a single command.

## Core features
- Reads a user's preprocessing logic in a multiline Python string (triple quotes string)
- Leakage-free preprocessing based on its fit/transform functionality
- Supports: mappings, one-hot encodings, row removals, column removals, outlier handling (IQR, percentile, z-score), imputations (KNeighbours, mode, median, mean), and normalization

## Internal logic and order of operations of Process Engine
- Applies row removals, column removals, mappings, one-hot encodings, and centralizes the format of invalid (missing) values to np.nan.
- Outlier handling. Anything that affects the distribution of data is dealt with before imputations or normalization
- Simple imputations. That is mean, median, mode.
- K-Nearest Neighbors (KNN) imputations based on the logic:  
  - Uses `KNeighborsClassifier` if the target column has 10 or fewer unique values; otherwise, uses `KNeighborsRegressor`.  
  - When fitting an imputer for a given column, all other columns called for KNN imputation and the target variable are excluded as inputs to ensure reliability of training  
  - Continuous columns used in KNN imputation are normalized before fitting the imputer. 
- Finally, normalization is applied.
- Each of the major operations above are encapsulated and separated by classes. Call them sub-preprocessors. Each sub-preprocessor has its own fit/transform functionality. The Processor class is the master class with the fit/transform that exists for the user. Its fit/transform is just the aggregate fit/transform of the sub-preprocessors (hidden classes)

## Simple Demonstration on thyroid cancer recurrence dataset
[Open In Colab](https://colab.research.google.com/drive/1UYE9HWP0FkJNhh5PH4tIbkSa0F_cOjHK#scrollTo=weY3Q6lN7jlT)
