import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, cross_validate


def kfold_validation(model, X, y, X_train, y_train, X_test, y_test):
    # 5 folds selected
    kfold = KFold(n_splits=5, random_state=0, shuffle=True)
    results = cross_val_score(model, X, y, cv=kfold)
    # Output the accuracy. Calculate the mean and std across all folds.
    print("Accuracy: %.3f%% (Std: %.3f%%)" %
          (results.mean()*100.0, results.std()*100.0))
    print('\n')
    model = model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    matrix = np.array(confusion_matrix(y_test, predictions))
    print(pd.DataFrame(matrix, index=['NO breakout', 'BREKOUT'], columns=[
        'Predicted_No_breakout', 'Predicted_BREAKOUT', ]))
    print('\n')
    print(classification_report(y_test, predictions))


def cross_validation(model, _X, _y, _cv=5):
    _scoring = ['accuracy', 'precision', 'recall', 'f1']
    results = cross_validate(estimator=model,
                             X=_X,
                             y=_y,
                             cv=_cv,
                             scoring=_scoring,
                             return_train_score=True)

    return {"Mean Training Accuracy": results['train_accuracy'].mean()*100,
            "Mean Training Precision": results['train_precision'].mean(),
            "Mean Training Recall": results['train_recall'].mean(),
            "Mean Training F1 Score": results['train_f1'].mean(),
            "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
            "Mean Validation Precision": results['test_precision'].mean(),
            "Mean Validation Recall": results['test_recall'].mean(),
            "Mean Validation F1 Score": results['test_f1'].mean()
            }


def crossValScores(model, X, y, X_train, y_train, X_test, y_test):
    scores = cross_val_score(model, X, y, cv=5)
    print("Scores mean : ", scores.mean()*100)
    print("\n")
    print("Accuracy : ", round(model.score(X_test, y_test)*100, 2))
    print("\n")
    predictions = model.predict(X_test)
    matrix = np.array(confusion_matrix(y_test, predictions))
    print(pd.DataFrame(matrix, index=['NO breakout', 'BREKOUT'], columns=[
        'Predicted_No_breakout', 'Predicted_BREAKOUT', ]))
    print('\n')
    target_names = ['No breakout', 'Breakout']
    print(classification_report(y_test, predictions, target_names=target_names))
    y_pred_proba = model.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
    auc = metrics.roc_auc_score(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label="data 1, auc="+str(auc))
    plt.legend(loc=4, prop={'size': 20})
    plt.show()
