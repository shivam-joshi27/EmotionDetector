import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# For serverless environments like Vercel, use /tmp for NLTK data
nltk_data_dir = os.environ.get("NLTK_DATA", "/tmp/nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir, exist_ok=True)
if nltk_data_dir not in nltk.data.path:
    nltk.data.path.append(nltk_data_dir)

# Lazy loading of NLTK resources
_lemmatizer = None
_stop_words = None

def download_nltk_resources():
    resources = ['wordnet', 'stopwords', 'punkt', 'punkt_tab', 'omw-1.4']
    for res in resources:
        try:
            nltk.data.find(f'corpora/{res}')
        except LookupError:
            try:
                nltk.data.find(f'tokenizers/{res}')
            except LookupError:
                nltk.download(res, download_dir=nltk_data_dir, quiet=True)

def get_lemmatizer():
    global _lemmatizer
    if _lemmatizer is None:
        download_nltk_resources()
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer

def get_stopwords():
    global _stop_words
    if _stop_words is None:
        try:
            _stop_words = set(stopwords.words('english'))
        except Exception:
            _stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're",
                "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he',
                'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's",
                'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which',
                'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
                'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
                'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because',
                'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
                'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no',
                'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
                'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll',
                'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't"
            }
    return _stop_words

def preprocess_text(text: str) -> str:
    """
    Full Text Preprocessing Pipeline:
    1. Converts text to lower case
    2. Removes URLs, special characters, numbers, and extra whitespaces
    3. Tokenizes text into words
    4. Removes English stopwords
    5. Lemmatizes tokens to their root form
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 1. Convert to lower case
    text = text.lower()
    
    # 2. Remove URLs, mentions, handles
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    
    # 3. Keep only alphabetic characters & spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 4. Tokenization
    try:
        tokens = word_tokenize(text)
    except Exception:
        tokens = text.split()
    
    # 5. Stopword removal & Lemmatization
    stop_words = get_stopwords()
    lemmatizer = get_lemmatizer()
    
    cleaned_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]
    
    return " ".join(cleaned_tokens)
