import pandas as pd
import numpy as np
import re
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import svm, metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

warnings.filterwarnings('ignore')
okgo = pd.read_csv('OKGO.csv', delimiter=";", skiprows=2, encoding='latin-1', engine='python') # read in the data
trump = pd.read_csv('trump.csv', delimiter=",", skiprows=2, encoding='utf-8', engine='python')
swift = pd.read_csv('TaylorSwift.csv', delimiter=",", skiprows=2, nrows=180, encoding='utf-8', engine='python')
royal = pd.read_csv('RoyalWedding.csv', delimiter=",", skiprows=2, nrows=61, encoding='utf-8', engine='python')
paul = pd.read_csv('LoganPaul.csv', delimiter=",", skiprows=2, nrows=200, encoding='utf-8', engine='python')
blogs = pd.read_csv('Kagel.csv', delimiter=",", skiprows=2, encoding='latin-1', engine='python') # read in the data
tweets = pd.read_csv('twitter.csv', delimiter=",", skiprows=2, encoding='latin-1', engine='python') # read in the data
tweets = tweets.drop(['Topic', 'TweetId', "TweetDate"], axis = 1).dropna()
tweets.head()

def fix_cols(DF):
    DF = DF.iloc[:,:2]
    DF.columns = ["label", "comment"]
    return DF

okgo = fix_cols(okgo)
trump = fix_cols(trump)
swift = fix_cols(swift)
royal = fix_cols(royal)
paul = fix_cols(paul)
tweets = fix_cols(tweets)
okgo.head()

tweets.label = tweets.label.replace({'positive': '1.0', 'negative':'-1.0', 'neutral': '0.0', 'irrelevant': '0.0'}, regex=True)
tweets['label'] = pd.to_numeric(tweets['label'], errors='coerce')

tweets = fix_cols(tweets)
blogs = fix_cols(blogs)

#tweets.head()
yt_comments = pd.concat([okgo, trump, swift, royal, paul], ignore_index=True)
#yt_comments.head()

non_yt_comments = pd.concat([blogs, tweets], ignore_index=True)
#non_yt_comments.head()

comments = pd.concat([yt_comments, non_yt_comments], ignore_index=True)
#comments.head()
def convert_to_string(DF):
    DF["comment"]= DF["comment"].astype(str)

convert_to_string(comments)

def cleanerFn(b):
    # keeps only words with alphabetic characters in comments
    for row in range(len(b)):
        line = b.loc[row, "comment"]
        b.loc[row,"comment"] = re.sub("[^a-zA-Z]", " ", line)

cleanerFn(comments)
nltk.download('stopwords')
sw = stopwords.words('english')
ps = PorterStemmer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def nlpFunction(DF):
    DF['com_token'] = DF['comment'].str.lower().str.split()
    DF['com_remv'] = DF['com_token'].apply(lambda x: [y for y in x if y not in sw])
    DF["com_lemma"] = DF['com_remv'].apply(lambda x : [lemmatizer.lemmatize(y) for y in x]) # lemmatization
    DF['com_stem'] = DF['com_lemma'].apply(lambda x : [ps.stem(y) for y in x]) # stemming
    DF["com_tok_str"] = DF["com_stem"].apply(', '.join)
    DF["com_full"] = DF["com_remv"].apply(' '.join)
    return DF

nltk.download('wordnet')
comments = nlpFunction(comments)


def drop_cols_after_nlp(comments):
    comments = comments.drop(columns = ['comment', 'com_token', 'com_remv', 'com_lemma', 'com_stem', 'com_tok_str'], axis = 1)
    return comments
comments = drop_cols_after_nlp(comments)
#comments.head()

comments.rename(columns = {'com_full': 'comment'}, inplace=True)
#comments.head()

def remove_missing_vals(comments):
    comments['comment'] = comments['comment'].str.strip()
    comments = comments[comments.comment != 'nan'] # remove nan values from data
    comments = comments[comments.comment != '']

remove_missing_vals(comments)

#comments.head()

comments['label'].isna().sum()

comments = comments[comments['label'].notna()]
comments['label'].isna().sum()

#len(comments)

X = comments['comment']
y = comments.label

# split X and y into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=53, test_size=0.25)
count_vectorizer = CountVectorizer(stop_words='english',
                                   min_df=0.05, max_df=0.9)

# Create count train and test variables
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)

# Initialize tfidf vectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english',
                                   min_df=0.05, max_df=0.9)

# Create tfidf train and test variables
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
# Create a MulitnomialNB model
tfidf_nb = MultinomialNB()
tfidf_nb.fit(tfidf_train,y_train)
# Run predict on your TF-IDF test data to get your predictions
tfidf_nb_pred = tfidf_nb.predict(tfidf_test)

# Calculate the accuracy of your predictions
tfidf_nb_score = metrics.accuracy_score(y_test,tfidf_nb_pred)

# Create a MulitnomialNB model
count_nb = MultinomialNB()
count_nb.fit(count_train,y_train)

# Run predict on your count test data to get your predictions
count_nb_pred = count_nb.predict(count_test)

# Calculate the accuracy of your predictions
count_nb_score = metrics.accuracy_score(count_nb_pred,y_test)

print('NaiveBayes Tfidf Score: ', tfidf_nb_score)
print('NaiveBayes Count Score: ', count_nb_score)

"""### Logistic Regression"""

lr_model = LogisticRegression()
lr_model.fit(tfidf_train,y_train)
accuracy_lr = lr_model.score(tfidf_test,y_test)
print("Logistic Regression accuracy is (for Tfidf) :",accuracy_lr)

lr_model = LogisticRegression()
lr_model.fit(count_train,y_train)
accuracy_lr = lr_model.score(count_test,y_test)
print("Logistic Regression accuracy is (for Count) :",accuracy_lr)
# # Create a SVM model

tfidf_svc = svm.SVC(kernel='linear', C=1)

tfidf_svc.fit(tfidf_train,y_train)
joblib.dump(tfidf_svc, 'saved_model.pkl')

# Run predict on your tfidf test data to get your predictions
tfidf_svc_pred = tfidf_svc.predict(tfidf_test)

# Calculate your accuracy using the metrics module
tfidf_svc_score = metrics.accuracy_score(y_test,tfidf_svc_pred)

print("LinearSVC Score (for tfidf):   %0.3f" % tfidf_svc_score)

count_svc = svm.SVC(kernel='linear', C=1)

count_svc.fit(count_train,y_train)
# Run predict on your count test data to get your predictions
count_svc_pred = count_svc.predict(count_test)

# Calculate your accuracy using the metrics module
count_svc_score = metrics.accuracy_score(y_test,count_svc_pred)

print("LinearSVC Score (for Count):   %0.3f" % tfidf_svc_score)

"""### Desicion Tree"""

from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier()
dt_model.fit(tfidf_train,y_train)
accuracy_dt = dt_model.score(tfidf_test,y_test)
print("Decision Tree accuracy is (for Tfidf):",accuracy_dt)

dt_model = DecisionTreeClassifier()
dt_model.fit(count_train,y_train)
accuracy_dt = dt_model.score(count_test,y_test)
print("Decision Tree accuracy is (for Count):",accuracy_dt)

"""### Random Forest"""

from sklearn.ensemble import RandomForestClassifier
rf_model_initial = RandomForestClassifier(n_estimators = 5, random_state = 1)
rf_model_initial.fit(tfidf_train,y_train)
print("Random Forest accuracy for 5 trees is (Tfidf):",rf_model_initial.score(tfidf_test,y_test))

rf_model_initial = RandomForestClassifier(n_estimators = 5, random_state = 1)
rf_model_initial.fit(count_train,y_train)
print("Random Forest accuracy for 5 trees is (Count):",rf_model_initial.score(count_test,y_test))
prediction_comments = pd.read_csv('output_comments.csv', delimiter=",", encoding='utf-8', engine='python')
prediction_comments = prediction_comments.iloc[:,:1]
prediction_comments.columns=['comment']
#prediction_comments.head()

# Lets use SVC to predict on our youtube video comments
#prediction_comments.head()

#len(prediction_comments['comment'])

convert_to_string(prediction_comments)
cleanerFn(prediction_comments)
prediction_comments = nlpFunction(prediction_comments)
prediction_comments = drop_cols_after_nlp(prediction_comments)
prediction_comments.rename(columns = {'com_full': 'comment'}, inplace=True)
remove_missing_vals(prediction_comments)
#prediction_comments.head()

tfidf_pred = tfidf_vectorizer.transform(prediction_comments['comment'])
tfidf_svc_pred = tfidf_svc.predict(tfidf_pred)

neutral = (tfidf_svc_pred == 0.0).sum()
positive = (tfidf_svc_pred == 1.0).sum()
negative = (tfidf_svc_pred < 0).sum()

print(neutral, positive, negative)

print("Good video" if positive > negative else "Bad video")