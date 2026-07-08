
import re
import string

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import contractions

# Make sure required NLTK data is available wherever this module is imported.
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

stemmer = PorterStemmer()
_STOPWORDS = set(stopwords.words('english'))


def preprocessing(text: str) -> str:
    """Cleans a single raw text string: lowercase, expand contractions,
    strip punctuation, tokenize, remove stopwords, and stem."""
    text = text.lower()
    text = contractions.fix(text)
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator)
    text = re.sub(r'\s{2,}', ' ', text)
    words = word_tokenize(text)
    y = [word for word in words if word not in _STOPWORDS]
    y = [stemmer.stem(word) for word in y]
    return ' '.join(y)


def preprocess_series(X):
    """Applies preprocessing() to every row of a pandas Series.

    Required because sklearn's FunctionTransformer passes the WHOLE Series
    into the wrapped function at once, not one string at a time. Without this
    wrapper, calling preprocessing(series) directly raises:
        AttributeError: 'Series' object has no attribute 'lower'
    """
    return X.apply(preprocessing)
