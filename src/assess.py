import pandas as pd
import numpy as np
import statistics
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    cross_validate,
    train_test_split,
    StratifiedKFold,
    cross_val_predict
)
from sklearn.metrics import roc_curve, RocCurveDisplay, auc
from itertools import cycle
from sklearn.preprocessing import LabelBinarizer

state = 123

# ' Calculate the five-fold cross-validation training and testing accuracies as well as the 
# ' possible range by mean and sd.
# '
# ' @param model: the Scikit-learn model or pipelie.
# ' @param X_train: X data in training set
# ' @param y_train: target values in training set
# ' @param **kwargs: any other parameters that can pass to the cross-validate function.
# '
# ' @return return a pandas Series containing the cross-validation results.
def mean_std_cross_val_scores(model, X_train, y_train, **kwargs):
    """
    Returns mean and std of cross validation
    """
    folds = StratifiedKFold(n_splits=5,random_state=state,shuffle=True)
    scores = cross_validate(model, X_train, y_train, **kwargs, cv=folds)

    mean_scores = pd.DataFrame(scores).mean()
    std_scores = pd.DataFrame(scores).std()
    out_col = []

    for i in range(len(mean_scores)):
        out_col.append((f"%0.3f (+/- %0.3f)" % (mean_scores[i], std_scores[i])))

    return pd.Series(data=out_col, index=mean_scores.index)


# ' calculate the 50% and 80% prediction interval.
# '
# ' @param ProbMatrix: The output prediction probability matrix by models.
# ' @param label: All classes in the response variable, default ascending order
# '
# ' @return Return a dictionary containing the prediction interval of 50% and 80% prediction intervals.

def CategoryPredInterval(ProbMatrix, labels):
    ncases = ProbMatrix.shape[0]
    pred50 = np.empty(ncases, dtype=object)
    pred80 = np.empty(ncases, dtype=object)

    for i in range(ncases):
        p = ProbMatrix[i, :]
        ip = np.argsort(-p)
        pOrdered = p[ip]
        labelsOrdered = np.array(labels)[ip]

        G = np.cumsum(pOrdered)
        k1 = min([k for k in range(len(G)) if G[k] >= 0.5])
        k2 = min([k for k in range(len(G)) if G[k] >= 0.8])

        pred1 = labelsOrdered[:k1+1].astype(str)
        pred2 = labelsOrdered[:k2+1].astype(str)

        pred50[i] = "".join(pred1)
        pred80[i] = "".join(pred2)

    return {'pred50': pred50, 'pred80': pred80}


# ' calculate the misclassification rate of 50% prediction interval and 80% prediction interval
# '
# ' @param pred_model: The predicted probability matrix by the model.
# ' @param y_test: Target calsses in the testing set.
# '
# ' @return Return a dictionary containing the misclassification rates based on prediction interval
# ' in 50% and 80% levels, and rates for each class.
def calc_misclass_rate_pred_interval(pred_model, y_test):
    res = CategoryPredInterval(pred_model, np.sort(y_test.unique()))
    res = pd.DataFrame(res)

    categories = np.sort(y_test.unique().astype(str))
    category_count_dict50 = {}
    category_total_dict50 = {}
    category_misrate_dict50 = {}
    category_count_dict80 = {}
    category_total_dict80 = {}
    category_misrate_dict80 = {}
    correct_count_50 = 0
    correct_count_80 = 0
    for cat in categories:
        category_count_dict50[cat] = 0
        category_total_dict50[cat] = 0
        category_misrate_dict50[cat] = 0
        category_count_dict80[cat] = 0
        category_total_dict80[cat] = 0
        category_misrate_dict80[cat] = 0

    for ind in range(len(y_test)):
        y = y_test.iloc[ind]
        rec50 = res['pred50'].iloc[ind]
        rec80 = res['pred80'].iloc[ind]
        category_total_dict50[str(y)] = category_total_dict50[str(y)] + 1
        category_total_dict80[str(y)] = category_total_dict80[str(y)] + 1
        if str(y) in rec50:
            correct_count_50 = correct_count_50 + 1
            category_count_dict50[str(y)] = category_count_dict50[str(y)] + 1
        if str(y) in rec80:
            correct_count_80 = correct_count_80 + 1
            category_count_dict80[str(y)] = category_count_dict80[str(y)] + 1

    for cat in categories:
        category_misrate_dict50[cat] = 1-category_count_dict50[cat]/category_total_dict50[cat]
        category_misrate_dict80[cat] = 1-category_count_dict80[cat]/category_total_dict80[cat]

    pred50 = statistics.mean(list(category_misrate_dict50.values()))
    pred80 = statistics.mean(list(category_misrate_dict80.values()))
    return {"pred50": pred50, "pred80": pred80, "class_pred50":category_misrate_dict50, "class_pred80":category_misrate_dict80,}


# ' Generate the confusion matrix for multi-nomial classification.
# '
# ' @param pred_model: The predicted probability matrix by the model.
# ' @param y_test: Target calsses in the testing set.
# '
# ' @return Return a table format confusion matrix.
def categoryPredMatrix(predModel, y_test):
    pred_df50 = pd.DataFrame()
    pred_df50['Smog Rating'] = y_test
    pred_df50['pred50'] = CategoryPredInterval(predModel, ["1", "3", "5", "6", "7"]).get('pred50')
    pred_df80 = pd.DataFrame()
    pred_df80['Smog Rating'] = y_test
    pred_df80['pred80'] = CategoryPredInterval(predModel, ["1", "3", "5", "6", "7"]).get('pred80')
    table50 = pd.pivot_table(
      data=pred_df50, 
      index='Smog Rating', 
      columns=['pred50'], 
      aggfunc=len, 
      fill_value=0
    )
    table80 = pd.pivot_table(
      data=pred_df80, 
      index='Smog Rating', 
      columns=['pred80'], 
      aggfunc=len, 
      fill_value=0
    )
    table = {'table50': table50, 'table80': table80}
    return table


# ' Helper function to calculate the misclassification rate of 50% prediction interval and 80% prediction interval
# ' for each cross-validation fold.
# '
# ' @param pred_model: The predicted probability matrix by the model.
# ' @param y_train_cv: Target calsses in the training set.
# '
# ' @return Return a dictionary containing the misclassification rates based on prediction interval
# ' in 50% and 80% levels for each cross-validation fold.
def calc_misclass_rate_pred_interval_cv(pred_model,y_train_cv):
    y_train_cv = pd.Series(y_train_cv)
    n_total = pred_model.shape[0]
    res = CategoryPredInterval(pred_model, np.sort(y_train_cv.unique()))
    res = pd.DataFrame(res)

    correct_count_50 = 0
    correct_count_80 = 0

    for ind in range(len(y_train_cv)):
        y = y_train_cv.iloc[ind]
        rec50 = res['pred50'].iloc[ind]
        rec80 = res['pred80'].iloc[ind]
        if str(y) in rec50:
            correct_count_50 = correct_count_50 + 1
        if str(y) in rec80:
            correct_count_80 = correct_count_80 + 1

    pred50 = correct_count_50/n_total
    pred80 = correct_count_80/n_total
    return {"pred50": round(1-pred50, 4), "pred80": round(1-pred80, 4)}


# ' calculate the misclassification rates for each cross validation.
# '
# ' @param pred_model: The predicted probability matrix by the model.
# ' @param X_train: The X data in the training set.
# ' @param y_train: The target values in the training set.
# ' @param **kwargs: any other parameters that can pass to the cross-validate function.
# '
# ' @return Return a list containing misclassification rates in each fold.
def cv_predict_interval(model, X_train, y_train, **kwargs):
    
    folds = StratifiedKFold(n_splits=5,random_state=state,shuffle=True)
    scores = cross_val_predict(model, X_train, y_train, **kwargs, cv=folds,method='predict_proba')
    probs = [scores[j] for i, j in folds.split(X_train,y_train)]
    y_train_new = y_train.values
    cv_ytrain = [y_train_new[j] for i,j in folds.split(X_train,y_train)]
    cv_pi = []
    for i in range(0, len(probs)):
        rate = calc_misclass_rate_pred_interval_cv(probs[i],cv_ytrain[i])
        cv_pi.append(rate)
    return cv_pi


# ' calculate the mean cross-validation misclassification rates.
# '
# ' @param array: The list containing cross-validation rates.
# '
# ' @return Return a dictionary containing mean values of cross-validation rates.
def get_mean_cv(array):
    preds_50 = []
    preds_80 = []
    for i in range(len(array)):
        fold = array[i]
        pred_50 = fold['pred50']
        pred_80 = fold['pred80']
        preds_50.append(pred_50)
        preds_80.append(pred_80)
    mean_50 = round(pd.DataFrame(preds_50).mean().values[0],3)
    mean_80 = round(pd.DataFrame(preds_80).mean().values[0],3)
    return {'cv_pred_50':mean_50, 'cv_pred_80':mean_80}



# ' Draw the ROC curve, and calculate the AUC scores.
# ' Reference: https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
# '
# ' @param model: The fitted pipeline or model.
# ' @param X_test: The X data in the testing set.
# ' @param y_train: The target values in the training set.
# ' @param y_test: The target values in the testing set.
# '
# ' @return Return a list containing AUC score for each class.
def auc_drawer(model, X_test, y_train, y_test):
  n_classes = len(np.unique(y_test))
  y_score = model.predict_proba(X_test)

  label_binarizer = LabelBinarizer().fit(y_train)
  y_onehot_test = label_binarizer.transform(y_test)
  fpr, tpr, roc_auc = dict(), dict(), dict()
  for i in range(n_classes):
      fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
      roc_auc[i] = auc(fpr[i], tpr[i])

  fpr_grid = np.linspace(0.0, 1.0, 1000)

  # Interpolate all ROC curves at these points
  mean_tpr = np.zeros_like(fpr_grid)

  for i in range(n_classes):
      mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

  # Average it and compute AUC
  mean_tpr /= n_classes

  fpr["macro"] = fpr_grid
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

  fig, ax = plt.subplots(figsize=(6, 6))

  plt.plot(
      fpr["macro"],
      tpr["macro"],
      label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
      color="navy",
      linestyle=":",
      linewidth=4,
  )

  target_names = np.unique(y_test)
  colors = cycle(["aqua", "darkorange", "cornflowerblue"])
  auc_list = {}
  for class_id, color in zip(range(n_classes), colors):
      RocCurveDisplay.from_predictions(
          y_onehot_test[:, class_id],
          y_score[:, class_id],
          name=f"ROC curve for {target_names[class_id]}",
          color=color,
          ax=ax,
      )

      auc_score = roc_auc_score(
          y_onehot_test[:, class_id],
          y_score[:, class_id]
      )
      auc_list[target_names[class_id]] = auc_score

  plt.plot([0, 1], [0, 1], "k--", label="ROC curve for chance level (AUC = 0.5)")
  plt.axis("square")
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("Extension of Receiver Operating Characteristic\nto One-vs-Rest multiclass")
  plt.legend()
  plt.show()
  return auc_list;


# ' Format the result table by testing results
# '
# ' @param dict: The dictionary of misclassification rates for each class.
# ' @param cmlist: The general confusion matrix in a list.
# ' @param acc: The test accuracy of the model.
# ' @param auclist: The list of AUC of each class.
# ' @param auc: The overall auc score.
# '
# ' @return Return a data frame of the results above.
def get_test_result_table(dict, cmlist,acc, auclist, auc):

    df = pd.DataFrame(dict).T
    pred50 = df.iloc[0,1]
    pred80 = df.iloc[1,1]
    df = df.iloc[2:,:]
    df['Overall'] = [pred50,pred80]
    cmlist.insert(len(cmlist),acc)
    auclist.insert(len(auclist),auc)
    df.loc['Accuracy'] = cmlist
    df.loc['AUC'] = auclist
    cmlist.remove(acc)
    auclist.remove(auc)
    return df.round(3)



# ' Format the result table by cross-validation results.
# '
# ' @param fold: The dictionary of misclassification rates for each cross-validation fold.
# ' @param mean_dict: The mean cross-validation rate..
# ' @param test: The validation accuracy.
# ' @param train: The training accuracy.
# '
# ' @return Return a data frame of the results above.
def cv_result_table(fold_dict, mean_dict, test, train):
    df = pd.DataFrame(fold_dict).round(3).T
    df.columns = ['1','2','3','4','5']
    mean_list = list(mean_dict.values())
    df['mean'] = mean_list
    train_list = ['-','-','-','-','-',train]
    test_list = ['-','-','-','-','-',test]
    df.loc['Train_Accuracy'] = train_list
    df.loc['Test_Accuracy'] = test_list
    return df