"""Text preparation, observable intermediate steps, and leakage-safe splitting."""
import re
import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
from sklearn.model_selection import train_test_split

LABELS = ['Natural Language Processing', 'Healthcare AI', 'Cybersecurity', 'Computer Vision', 'Robotics']
STEMMER = PorterStemmer()

def tokens(text):
    return re.findall(r'[a-z]+', str(text).casefold())

def preprocess_steps(text):
    tokenized = re.findall(r'[A-Za-z]+', str(text))
    folded = [w.casefold() for w in tokenized]
    filtered = [w for w in folded if w not in ENGLISH_STOP_WORDS]
    stemmed = [STEMMER.stem(w) for w in filtered]
    return {'original': text, 'tokenization': tokenized, 'case_folding': folded,
            'stopword_removal': filtered, 'stemming': stemmed, 'processed': ' '.join(stemmed)}

def prepare_data(path, seed=42):
    df = pd.read_csv(path)
    required = ['paper_id', 'title', 'abstract', 'research_domain']
    if not set(required).issubset(df.columns):
        raise ValueError(f'Expected columns: {required}; found {list(df.columns)}')
    if df[required].isna().any().any():
        raise ValueError('Missing required values; inspect the CSV before continuing.')
    if set(df.research_domain) != set(LABELS):
        raise ValueError(f'Unexpected labels: {set(df.research_domain)}')
    df['text'] = df.title.astype(str) + '. ' + df.abstract.astype(str)
    df['_key'] = df.text.str.casefold().str.replace(r'\s+', ' ', regex=True).str.strip()
    if (df.groupby('_key').research_domain.nunique() > 1).any():
        raise ValueError('Identical texts have conflicting labels.')
    before = len(df)
    df = df.drop_duplicates('_key').reset_index(drop=True)
    df['processed'] = df.text.map(lambda x: preprocess_steps(x)['processed'])
    train, test = train_test_split(df, test_size=0.2, random_state=seed, stratify=df.research_domain)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    assert set(train._key).isdisjoint(set(test._key))
    print(f'Loaded {before} rows; removed {before-len(df)} exact text duplicates. Train={len(train)}, test={len(test)}')
    return df, train, test

def bag_of_words(train, test):
    vectorizer = CountVectorizer()
    x_train = vectorizer.fit_transform(train.processed)
    x_test = vectorizer.transform(test.processed)
    print(f'Bag of Words: train={x_train.shape}, test={x_test.shape}; vocabulary fitted on training only.')
    return vectorizer, x_train, x_test
