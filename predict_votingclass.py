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
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

def cv(x_train, y_train):
    md = np.arange(3, 26)
    scores = [] 
    for maxDepth in md: 
        print("clf:", maxDepth)
        clf = DecisionTreeClassifier(max_depth=maxDepth)
        score = cross_val_score(clf, x_train, y_train, cv=10, scoring='accuracy') #10 partions/fold cross validation
        scores.append(score.mean())
        scores
    # plot to see clearly
    plt.plot(md, scores)
    plt.xlabel('Value of MaxDepth for Decision Tree Classifier')
    plt.ylabel('Cross-Validated Accuracy')
    plt.show()

    max_value = max(scores)
    idx = scores.index(max_value)
    print("Best accuracy score with max depth of: ", md[idx])
    return md[idx]

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
X_train_processed = X_train.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'stemmed_summary', 'Date', 'Hour', 'HelpfulnessNumerator','HelpfulnessDenominator', 'Tier', 'all','an','and','as','at','be', 'do', 'for', 'from','the', 'so'])
X_test_processed = X_test.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'stemmed_summary', 'Date', 'Hour', 'Score', 'HelpfulnessNumerator','HelpfulnessDenominator', 'Tier', 'all','an','and','as','at','be', 'do', 'for', 'from','the', 'so'])
X_submission_processed = X_submission.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'stemmed_summary', 'Date', 'Hour', 'Score', 'HelpfulnessNumerator','HelpfulnessDenominator', 'Tier', 'all','an','and','as','at','be', 'do', 'for', 'from','the', 'so'])

#Under Sample the Training Set
def under_sample(df, col_name): 
        five = df[df[col_name]==5.0]
        four = df[df[col_name]==4.0]
        three = df[df[col_name]==3.0]
        two = df[df[col_name]==2.0]
        one = df[df[col_name]==1.0]
        five = five.sample(n=len(four), random_state=0) #undersample count (Score =5) to count (Score =1)
        df = pd.concat([five,four, three, two, one],axis=0) #concat rows 
        return df
#X_train_processed = under_sample(X_train_processed, 'Score')
#X_train_processed.sample(frac = 0.10, random_state=0)
y_train_processed = X_train_processed['Score']
X_train_processed  = X_train_processed.drop(columns=['Score']) # drop score, don't need it anymore 
print(X_train_processed.head())

# Get hyperparameter k CV Score and learn the model
clf1 = DecisionTreeClassifier(max_depth=12)   #decision tree
clf2 = KNeighborsClassifier(n_neighbors=10)   #kneighbors   
clf3 = SVC(kernel='linear',probability=True) #support vector machine 
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2),
                                    ('svc', clf3)],
                        voting='soft', 
                        weights=[1, 1, 2]) #the weights determine how importance each classifier's vote is
print('fitting..')
eclf.fit(X_train_processed,y_train_processed)
print('done fitting')

# Predict the reponse for the test dataset 
y_predict = eclf.predict(X_test_processed)
accuracy_score(y_predict, Y_test)

# Predict the score using the model
Y_test_predictions = eclf.predict(X_test_processed)
X_submission['Score'] = eclf.predict(X_submission_processed)

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