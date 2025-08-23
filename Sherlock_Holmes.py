#Sherlock_Holmes.py
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
import joblib

def Sherlock_Holmes():
    
    tfidf = TfidfVectorizer(ngram_range=(1,1), max_features=2000)
    
    
    ngrams = CountVectorizer(ngram_range=(2,2))
    
    
    features = FeatureUnion([
        ('tfidf', tfidf),
        ('ngrams', ngrams)
    ])
    
    
    pipeline = Pipeline([
        ('features', features),
        ('clf', MultinomialNB(alpha=1.0))  
    ])
    
    return pipeline