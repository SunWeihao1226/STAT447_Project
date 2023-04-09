STAT447_Project

Purpose for each file:

under root directory:

1. 01_data_cleaning.ipynb: contains all relevant code blocks in data cleaning, feature selection, data transformation and data wrangling.
2. 02_gktau.ipynb: first binning all numerical variables to ordinal categorical variables, then calculate the sqrt(gktau) for each explanatory variable in predicting response variable and generate a table.
3. 03a_random_forest.ipynb: contains the relavant code blocks of random forest model and weighted random forest model.
4. 03b_naive_bayes.ipynb: contains the relavant code blocks of Gaussian Naive Bayes model and Gaussian Naive Bayes model with PCA.
5. 03c_logistic_regression.ipynb: contains the relavant code blocks of multinomial logistic model, weighted multinomial logistic model and weighted multinomial logistic model with interaction term
6. 03d_knn.ipynb: contains the relavant code blocks of KNN model with tunned with Gridsearch
7. 03e_svc.ipynb: contains the relavant code blocks of SVC model with tunned with Gridsearch
8. smog_data.csv: The original dataset
9. smog_data_cleaned.csv: The processed dataset after data cleaning


under directory /src:

1. assess.py: contains all helper functions that will be frequently used by the other files
2. gktau.py: contains the reveleant functions in calculating sqrt(gktau)
3. lr_interaction.py: contains the relevant helper functions for preparing the data used for weighted multinomial logistic regression with interaction model. A dictionary aims at transferring one-hot-encoding index to corresponding column names is also included.
4. pklreader.py: contains a helper function used for reading those stored workspace files from `pickle` library in python
5. preprocessor.py: contains a helper function initializing one-hot-encoder for categorical and binary variables and StandardScaler for numerical variables


under directory /workspace_file:

1. 01a_datacleaning_summary.pkl: The summary table of numerical variables(without Fuel Consumption(Comb (mpg) and year)
2. 01b_smog_data_cleaned.pkl: The dataset after data cleaning
3. 02_gktau_tb.pkl: The sqrt(gktau) table for each explanatory variable vs smog rating
4. 03a_1_rf_cv.pkl: The cross-validation result of random forest model(train/test(validation set) accuracy, 50% and 80% in-sample misclassification rate)
5. 03a_2_rf_test.pkl: The test set result of random forest model(50% and 80% out-of-sample misclassification rate, test accuracy and macro-averaged AUC score)
6. 03a_3_wrf_cv.pkl: The cross-validation result of weighted random forest model(train/test(validation set) accuracy, 50% and 80% in-sample misclassification rate)
7. 03a_4_wrf_test.pkl: The test set result of weigted random forest model(50% and 80% out-of-sample misclassification rate, test accuracy and macro-averaged AUC score)
8. 03a_5_rfmodel.pkl: The pipeline of the unweighted random forest model.
9. 03a_5_rfmodel.pkl: The pipeline of the unweighted random forest model.
10. 03b_1_nb_cv.pkl: The cross-validation result of Gaussian Naive Bayes model(train/test(validation set) accuracy, 50% and 80% in-sample misclassification rate)
11. 03b_2_nb_test.pkl: The test set result of Gaussian Naive Bayes model(50% and 80% out-of-sample misclassification rate, test accuracy and macro-averaged AUC score)
12. 03b_3_nb_pca_cv.pkl: The cross-validation result of Gaussian Naive Bayes model with PCA(train/test(validation set) accuracy, 50% and 80% in-sample misclassification rate)
13. 03b_4_nb_pca_test.pkl: The test set result of Gaussian Naive Bayes model with PCA(50% and 80% out-of-sample misclassification rate, test accuracy and macro-averaged AUC score)
14. 03b_5_nbmodel.pkl: The pipeline of the Naive Bayes model.
15. 03b_6_nb_pcamodel.pkl: The pipeline of the Naive Bayes model for the data with PCA.
16. 03c_1_lr_cv.pkl: The cross-validation result of multinomial logistic regression model(train/test(validation set) accuracy, 50% and 80% in-sample misclassification rate)
17. 03c_2_lr_test.pkl: The test set result of multinomial logistic regression model(50% and 80% out-of-sample misclassification rate, test accuracy and macro-averaged AUC score)
18. 03c_3_wlr_cv.pkl: The cross-validation result of weighted multinomial logistic regression model(train/test(validation set) accuracy, 50% and 80% in-sample misclassification rate)
19. 03c_4_wlr_test.pkl: The test set result of weighted multinomial logistic regression model(50% and 80% out-of-sample misclassification rate, test accuracy and macro-averaged AUC score)
20. 03c_5_intr_cv.pkl: The cross-validation result of weighted multinomial logistic regression model with interaction(train/test(validation set) accuracy, 50% and 80% in-sample misclassification rate)
21. 03c_6_intr_test.pkl: The test set result of weighted multinomial logistic regression model with interaction(50% and 80% out-of-sample misclassification rate, test accuracy and macro-averaged AUC score)
22. 03c_7_lrmodel.pkl: The pipeline of the unweighted multinomial logistic regression model.
23. 03c_8_wlrmodel.pkl: The pipeline of the weighted multinomial logistic regression model.
24. 03c_9_intrmodel.pkl: The pipeline of the weighted multinomial logistic regression model with interaction term.
25. 03d_1_knn_cv.pkl: The cross-validation result of KNN(train/test(validation set) accuracy, 50% and 80% in-sample misclassification rate)
26. 03d_2_knn_test.pkl: The test set result of KNN(50% and 80% out-of-sample misclassification rate, test accuracy and macro-averaged AUC score)
27. 03d_3_knn_model.pkl: The pipeline of the KNN model.
28. 03e_1_svc_cv.pkl: The cross-validation result of SVC(train/test(validation set) accuracy, 50% and 80% in-sample misclassification rate)
29. 03e_2_svc_test.pkl: The test set result of SVC(50% and 80% out-of-sample misclassification rate, test accuracy and macro-averaged AUC score)
30. 03e_3_svc_model.pkl: The pipeline of the SVC model.



File Running Order:
1. 01_data_cleaning.ipynb. 
2. 02_gktau.ipynb
3. 03a_random_forest.ipynb
4. 03b_naive_bayes.ipynb
5. 03c_logistic_regression.ipynb
6. 03d_knn.ipynb
7. 03e_svc.ipynb


Libraries and function are used:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import numpy as np
import dataframe_image as dfi
import pickle
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.utils import class_weight
from sklearn.metrics import roc_auc_score, confusion_matrix