from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score

def classification_metrics(y_true, y_pred):
  
    print("Precision score: {}".format(precision_score(y_true,y_pred)))
    print("Recall score: {}".format(recall_score(y_true,y_pred)))
    print("F1 Score: {}".format(f1_score(y_true,y_pred)))
    print("Confusion Matrix \n" )
    print(confusion_matrix(y_true, y_pred))