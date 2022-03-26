import pandas as pd
import numpy as np
import matplotlib as plt 
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
snow_stemmer = SnowballStemmer(language='english')
sia = SentimentIntensityAnalyzer()

nltk.download([
     "names",
     "stopwords",
     "state_union",
     "twitter_samples",
     "movie_reviews",
     "averaged_perceptron_tagger",
     "vader_lexicon",
     "punkt", ])

nltk.download('stopwords')
SEED = 2020

def z_score(df):
    """
    This function calculates the z-score for the Fare column 

    df: a pd dataframe containing the titantic dataset 

    return a new column with z-scores
    """
    return (df.mean() - df)/ df.std()
def process(df):
    # This is where you can do all your processing
    print("This is where you can do all your processing")
    df['Helpfulness'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']
    df['Helpfulness'] = df['Helpfulness'].fillna(0)
    df['Summary'] = df['Summary'].fillna('')
    df['Text'] = df['Text'].fillna('')
    df['Helpfulness'] = z_score(df['Helpfulness']) #normalize helpfulness
    df['Score'].fillna(df['Score'].median(), inplace = True) #median is best estimate for center of average in skewed data 
    #df = df.dropna()
    df['Summary'] = df['Summary'].str.lower()
    df['Text'] = df['Text'].str.lower()
    df['Date'] = pd.to_datetime(df['Time'], unit = 's')
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    df['Hour'] = df['Date'].dt.hour
    df['ReviewLength'] = df.apply(lambda row : len(row['Text'].split()) if type(row['Text']) == str else 0, axis = 1)
    df['SummaryLength'] = df.apply(lambda row : len(row['Summary'].split()) if type(row['Summary']) == str else 0, axis = 1)
    df['ReviewLength_Char'] = df['Text'].str.len()
    df['SummaryLength_Char'] = df['Summary'].str.len()
    df['Review_Ratio'] = df['ReviewLength_Char'] / df['ReviewLength'] #takes the avg number of char length for each word in a row for review 
    df['Summary_Ratio'] = df['SummaryLength_Char'] / df['SummaryLength'] #takes the avg number of char length for each word in a row for summary 
    df["Spec_Char"] = df['Text'].str.findall(r'[^a-zA-Z0-9 ]').str.len() #find all special characters and count them 
    df['Review_Ratio'] = df['Review_Ratio'].fillna(0)
    df['Summary_Ratio'] = df['Summary_Ratio'].fillna(0)
    print(df.isna().sum(axis = 0)) #check to make sure no NaNs

    def tokenize_it(data): 
        stemmed_data = []
        stemmed_data = [" ".join(SnowballStemmer("english", ignore_stopwords=True).stem(word)
                    for sent in sent_tokenize(message)
                        for word in word_tokenize(sent))
                        for message in data]
        return stemmed_data
    print("Tokenizing...")
    df['stemmed_summary'] = tokenize_it(df['Summary'])
    print("Sentiment Scores...")
    df['sentiment_summary'] = df['stemmed_summary'].apply(lambda x: sia.polarity_scores(x)['compound'])
    #df['sentiment_summary'] = df['sentiment_summary'].fillna(df['sentiment_summary'].mean())
    print("Vectorizer step...")
    vectorizer = TfidfVectorizer(max_df = .55, min_df = 0.01) #filter out words that occur in 98% articles and that don't occur in 1% articles
    x = vectorizer.fit_transform(df['stemmed_summary']) #fit and transform all words in stemmed/tokenized sumamry 
    df1 = pd.DataFrame(x.toarray(), columns=vectorizer.get_feature_names())
    res = pd.concat([df, df1], axis=1)

    print("Creating Binary Classifiers...")
    # create a list of our conditions
    conditions = [
        (res['Score'] <= 2),
        (res['Score'] > 2) ]
    values = [0, 1]
    res['Tier'] = np.select(conditions, values) #0 is score less than 

    # create a list of our conditions
    conditions = [
        (res['sentiment_summary'] <= 0),
        (res['sentiment_summary'] > 0) ]
    values = [0, 1]
    res['Polarity'] = np.select(conditions, values)

    # create a list of our conditions
    conditions = [
        (res['Helpfulness'] <= 0),
        (res['Helpfulness'] > 0) ]
    values = [0, 1]
    res['HelpBin'] = np.select(conditions, values)
    print("DONE")
    return res



# Load the dataset
trainingSet = pd.read_csv("./data/train.csv")

# Process the DataFrame
train_processed = process(trainingSet)

# Load test set
submissionSet = pd.read_csv("./data/test.csv")

# Merge on Id so that the test set can have feature columns as well
testX= pd.merge(train_processed, submissionSet, left_on='Id', right_on='Id')
testX = testX.drop(columns=['Score_x'])
testX = testX.rename(columns={'Score_y': 'Score'})

# The training set is where the score is not null
trainX =  train_processed[train_processed['Score'].notnull()]

testX.to_csv("./data/X_test.csv", index=False)
trainX.to_csv("./data/X_train.csv", index=False)

