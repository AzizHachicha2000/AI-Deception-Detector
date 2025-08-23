#preprocessor.py
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download(['punkt', 'stopwords', 'wordnet', 'omw-1.4', 'punkt_tab'])

def preprocess_text(text):
    
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    
    
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    
    
    stops = set(stopwords.words('english')) - {'no', 'not', 'never'}
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stops]
    
    return ' '.join(tokens)