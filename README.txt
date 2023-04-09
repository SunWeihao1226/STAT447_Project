STAT447_Project

Purpose for each file:

under root directory

1. 01_data_cleaning.ipynb: contains all relevant code blocks in data cleaning, feature selection, data transformation and data wrangling.
2. 02_gktau.ipynb: first binning all numerical variables to ordinal categorical variables, then calculate the sqrt(gktau) for each explanatory variable in predicting response variable and generate a table.
3. 03a_random_forest.ipynb: contains the relavant code blocks of random forest model and weighted random forest model.
4. 03b_naive_bayes.ipynb: contains the relavant code blocks of Gaussian Naive Bayes model and Gaussian Naive Bayes model with PCA.
5. 03c_logistic_regression.ipynb: contains the relavant code blocks of multinomial logistic model, weighted multinomial logistic model and weighted multinomial logistic model with interaction term
6. 03d_knn.ipynb: contains the relavant code blocks of KNN model with tunned with Gridsearch
7. 03e_svc.ipynb: contains the relavant code blocks of SVC model with tunned with Gridsearch
8. smog_data.csv: The original dataset
9. smog_data_cleaned.csv: The processed dataset after data cleaning

under directory /src

1. assess.py: contains all helper functions that will be frequently used by the other files
2. gktau.py: contains the reveleant functions in calculating sqrt(gktau)
3. lr_interaction.py: contains the relevant helper functions for preparing the data used for weighted multinomial logistic regression with interaction model. A dictionary aims at transferring one-hot-encoding index to corresponding column names is also included.
4. pklreader.py: contains a helper function used for reading those stored workspace files from `pickle` library in python
5. preprocessor.py: contains a helper function initializing one-hot-encoder for categorical and binary variables and StandardScaler for numerical variables
