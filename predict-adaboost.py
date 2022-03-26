from tkinter import Y
import pandas as pd
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn import under_sampling
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

def cv(x_train, y_train):
    scores = [] 
    n_trees = [10, 50, 100, 500, 1000, 1500]
    for n in n_trees: 
        print("Trees Conisdering:", n)
        model = AdaBoostClassifier(n_estimators=n)
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
        score = cross_val_score(model, x_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
        scores.append(score.mean())
    # plot to see clearly
    plt.plot(n_trees, scores)
    plt.xlabel('Value of MaxDepth for Decision Tree Classifier')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()

    max_value = max(scores)
    idx = scores.index(max_value)
    print("Best accuracy score with max depth of: ", n_trees[idx])
    return n_trees[idx]

# Load files into DataFrames
X_train = pd.read_csv("./data/X_train.csv")
X_submission = pd.read_csv("./data/X_test.csv")

# Split training set into training and testing set
X_train, X_test, Y_train, Y_test = train_test_split(
        X_train,
        X_train['Score'],
        test_size=1/4.0,
        random_state=0
    )

X_train_processed = X_train.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'stemmed_summary', 'Date', 'Hour', 'Tier', 'all','an','and','as','at','be','best','better','but','classic','do','dvd','ever','film','for','from','get','have','in','is','it','just','like','love','more','movi','my','not','of','on','one','review','season','seri','show','so','star','stori','than','that','the','this','time','to','very','was','watch','what','with','you'])
X_test_processed = X_test.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'stemmed_summary', 'Date', 'Hour', 'Score', 'Tier', 'all','an','and','as','at','be','best','better','but','classic','do','dvd','ever','film','for','from','get','have','in','is','it','just','like','love','more','movi','my','not','of','on','one','review','season','seri','show','so','star','stori','than','that','the','this','time','to','very','was','watch','what','with','you'])
X_submission_processed = X_submission.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'stemmed_summary', 'Date', 'Hour', 'Score', 'Tier', 'all','an','and','as','at','be','best','better','but','classic','do','dvd','ever','film','for','from','get','have','in','is','it','just','like','love','more','movi','my','not','of','on','one','review','season','seri','show','so','star','stori','than','that','the','this','time','to','very','was','watch','what','with','you'])

#Under Sample the Training Set
def under_sample(df, col_name): 
        five = df[df[col_name]==5.0]
        four = df[df[col_name]==4.0]
        three = df[df[col_name]==3.0]
        two = df[df[col_name]==2.0]
        one = df[df[col_name]==1.0]
        five = five.sample(n=len(one), random_state=0) #undersample count (Score =5) to count (Score =1)
        df = pd.concat([five,four, three, two, one],axis=0) #concat rows 
        return df
#X_train_processed = under_sample(X_train_processed, 'Score')
y_train_processed = X_train_processed['Score']
X_train_processed  = X_train_processed.drop(columns=['Score']) # drop score, don't need it anymore 
print(X_train_processed.head())

# Get hyperparameter k CV Score and learn the model
#n_trees = cv(X_train_processed, y_train_processed)
model = AdaBoostClassifier(n_estimators=1500).fit(X_train_processed, y_train_processed)

# Predict the reponse for the test dataset 
y_predict = model.predict(X_test_processed)
accuracy_score(y_predict, Y_test)

# Predict the score using the model
Y_test_predictions = model.predict(X_test_processed)
X_submission['Score'] = model.predict(X_submission_processed)

# Evaluate your model on the testing set
print("Accuracy on testing set = ", accuracy_score(Y_test, Y_test_predictions))

# Plot a confusion matrix
cm = confusion_matrix(Y_test, Y_test_predictions, normalize='true')
sns.heatmap(cm, annot=True)
plt.title('Confusion matrix of the classifier')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Create the submission file
submission = X_submission[['Id', 'Score']]
submission.to_csv("./data/submission.csv", index=False)