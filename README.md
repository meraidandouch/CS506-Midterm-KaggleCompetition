# CS506 Midterm


## .Py Files Description
This github repo attempts to predict Amazon review scores by analyzing and modeling 1,697,533 unique reviews from the Amazon Movie Review dataset, with their associated star ratings (Score) and metadata. Variables in the metadata included ProductId, UserId, HelpfulnessNumerator, HelpfulnessDenominator, Score, Time, Summary, Text, and Id. The problem lies in that a large percentage of the reviews are 5 stars introducing bias in the data. Feature extraction, machine learning algorithms such as Decision Tree Classifier are implemented and tested using cross model validation to increase the model accuracy on the testing dataset. 

1. **feature_extraction.py** - this file extracts important features from the dataset such as sentiment analysis (polarity), and feature names using TFIDF vectorizer machine
    * Important Features: sentiment_summary', 'Helpfulness', 'great', 'bad', 'better', 'funni', 'time' 
3. **predict-SVC.py** - apply SVC machine learning method on training/testing set obtained from feature_extraction.py
4. **predict-adaboost.py** - apply adaboost machine learning method on training/testing set obtained from feature_extraction.py
5. **predict-clf.py** - apply decision tree classifier machine learning method on training/testing set obtained from feature_extraction.py
6. **predict-constant.py** - provided by professor to test submission on kaggle
7. **predict-knn.py** - apply KNN machine learning method on training/testing set obtained from feature_extraction.py
8. **predict_votingclass.py** - apply votingclass machine learning method on training/testing set obtained from feature_extraction.py
9. **requirements.txt** - provided by professor to test submission on kaggle
10. **test_setup.py** - provided by professor to test submission on kaggle
11. **visualize.ipynb** - jupyter notebook for preliminary analysis
