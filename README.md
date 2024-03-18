# Computational Data Analysis Project: Predictive Modeling with Missing Data Imputation

## Overview
This project is part of the Computational Data Analysis course, aimed at building a predictive model for a continuous response variable `Y` based on 100 features from 100 observations. The dataset presents a mix of continuous and categorical variables, with a significant portion of missing values. The challenge involves data imputation, model selection, validation, and estimating prediction error.

## Data Description
The dataset comprises 100 observations with 100 features, of which 90 are continuous and the rest are categorical. It includes a notable 15% missing data across these features, necessitating careful imputation strategies.

## Files
- `case1Data.txt` & `case1DataXnew.txt`: Data files with original and new observations.
- `predictionsYourStudentNos.txt`: Predictions on the new data.
- `estimatedRMSEYourStudentNos.txt`: Estimated RMSE of the predictions.

## Methodology
1. **Data Normalization and Encoding**: Continuous variables were normalized, and categorical variables were one-hot encoded.
2. **Missing Data Imputation**: Utilized a Gaussian Mixture Model (GMM) approach after assessing the data's distribution, supplemented by initial KNN imputation.
3. **Model Selection**: An Elastic Net regression model was chosen due to its aptitude for handling highly correlated variables and datasets with more features than observations.
4. **Validation**: Model performance was validated using a split of 80% training data, with hyperparameters optimized through 5-fold grid search.

## Results
- The Elastic Net model achieved an RMSE of 22.37 on the validation set and 14.04 on the entire dataset, indicating reasonable predictive performance.
- The residuals analysis suggested a generally good fit, with some signs of heteroscedasticity.
  
## Contributors
- Lucas Brylle
- Mads Prip

```
