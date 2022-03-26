from tkinter import Y
import pandas as pd
import seaborn as sns
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn import under_sampling
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

def cv(x_train, y_train):
    md = np.arange(3, 24)
    scores = [] 
    for maxDepth in md: 
        print("clf:", maxDepth)
        clf = MultinomialNB()
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
#X_train_processed = X_train.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'stemmed_summary', 'Date', 'Hour', 'Tier', 'HelpfulnessDenominator', 'all','an','and','as','at','be','best','but','classic','do','ever','film','for','from','get','good','have','in','is','it','just','like','love','more','movi','my','not','of','on','one','review','season','show','so','star','stori','than','that','the','this','time','to','very','was','watch','what','with','you'])
#X_test_processed = X_test.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'stemmed_summary', 'Date', 'Hour', 'Score', 'Tier', 'HelpfulnessDenominator', 'all','an','and','as','at','be','best','but','classic','do','ever','film','for','from','get','good','have','in','is','it','just','like','love','more','movi','my','not','of','on','one','review','season','show','so','star','stori','than','that','the','this','time','to','very','was','watch','what','with','you'])
#X_submission_processed = X_submission.drop(columns=['Id', 'ProductId', 'UserId', 'Text', 'Summary', 'stemmed_summary', 'Date', 'Hour', 'Score', 'Tier', 'HelpfulnessDenominator', 'all','an','and','as','at','be','best','but','classic','do','ever','film','for','from','get','good','have','in','is','it','just','like','love','more','movi','my','not','of','on','one','review','season','show','so','star','stori','than','that','the','this','time','to','very','was','watch','what','with','you'])
X_train_processed = X_train[['Score', 'Polarity', 'HelpBin', 'Spec_Char', 'ReviewLength','good', 'bad', 'great']]
X_test_processed = X_test[['Polarity', 'HelpBin', 'Spec_Char', 'ReviewLength','good','bad', 'great']]
X_submission_processed = X_submission[['Polarity', 'HelpBin', 'Spec_Char', 'ReviewLength', 'good', 'bad', 'great']]


#Under Sample the Training Set
def under_sample(df, col_name): 
        five = df[df[col_name]==5.0]
        four = df[df[col_name]==4.0]
        three = df[df[col_name]==3.0]
        two = df[df[col_name]==2.0]
        one = df[df[col_name]==1.0]
        five = five.sample(n=len(two), random_state=0) #undersample all classes to len of lowest class(Score =2) to count (Score =1)
        four = four.sample(n=len(two), random_state=0) 
        three = three.sample(n=len(two), random_state=0) 
        one = one.sample(n=len(two), random_state=0)
        df = pd.concat([five,four, three, two, one],axis=0) #concat rows 
        return df
X_train_processed = under_sample(X_train_processed, 'Score')
y_train_processed = X_train_processed['Score']
X_train_processed  = X_train_processed.drop(columns=['Score']) # drop score, don't need it anymore 
print(X_train_processed.head())

# Get hyperparameter k CV Score and learn the model
#md = cv(X_train_processed, y_train_processed)
model = MultinomialNB().fit(X_train_processed, y_train_processed)

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