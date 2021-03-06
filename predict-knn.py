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
from sklearn.tree import DecisionTreeClassifier

#import imblearn
#from imblearn.pipeline import Pipeline
#from imblearn.over_sampling import RandomOverSampler

def cv(train_x, train_y): 
    scores = []
    k_range = np.arange(1,21)
    for i in range(1,21): 
        print("KNN" + str(i))
        KNN = KNeighborsClassifier(n_neighbors=i)

        # X,y will automatically be divided into 10 parts, the scoring I will still use the accuracy
        score = cross_val_score(KNN, train_x, train_y, cv=10, scoring='accuracy') #10 partions/fold cross validation
        scores.append(score.mean())

    # plot to see clearly
    plt.plot(k_range, scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()
    max_value = max(scores)
    idx = scores.index(max_value)
    print("Best accuracy score with max depth of: ", k_range[idx])
    return k_range[idx]

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

# This is where you can do more feature selection
X_train_processed = X_train.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'stemmed_summary', 'Date', 'Hour', 'Tier', 'all','an','and','as','at','be','best','better','but','classic','do','dvd','ever','film','for','from','get','good','have','in','is','it','just','like','love','more','movi','my','not','of','on','one','review','season','seri','show','so','star','stori','than','that','the','this','time','to','very','was','watch','what','with','you'])
X_test_processed = X_test.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'stemmed_summary', 'Date', 'Hour', 'Score', 'Tier', 'all','an','and','as','at','be','best','better','but','classic','do','dvd','ever','film','for','from','get','good','have','in','is','it','just','like','love','more','movi','my','not','of','on','one','review','season','seri','show','so','star','stori','than','that','the','this','time','to','very','was','watch','what','with','you'])
X_submission_processed = X_submission.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'stemmed_summary', 'Date', 'Hour', 'Score', 'Tier', 'all','an','and','as','at','be','best','better','but','classic','do','dvd','ever','film','for','from','get','good','have','in','is','it','just','like','love','more','movi','my','not','of','on','one','review','season','seri','show','so','star','stori','than','that','the','this','time','to','very','was','watch','what','with','you'])

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
#k = cv(X_train_processed, y_train_processed)
model = DecisionTreeClassifier(max_depth=9).fit(X_train_processed, y_train_processed)

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