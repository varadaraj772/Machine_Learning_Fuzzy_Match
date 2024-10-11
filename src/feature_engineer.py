from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # Check for required columns
    if 'string_1' not in df or 'string_2' not in df:
        raise ValueError("DataFrame must contain 'string_1' and 'string_2' columns.")

    # Compute Levenshtein and partial ratios
    df['levenshtein_ratio'] = df.apply(lambda x: fuzz.ratio(x['string_1'], x['string_2']), axis=1)
    df['partial_ratio'] = df.apply(lambda x: fuzz.partial_ratio(x['string_1'], x['string_2']), axis=1)
    
    # Additional fuzzy features
    df['token_set_ratio'] = df.apply(lambda x: fuzz.token_set_ratio(x['string_1'], x['string_2']), axis=1)
    df['token_sort_ratio'] = df.apply(lambda x: fuzz.token_sort_ratio(x['string_1'], x['string_2']), axis=1)

    # Vectorize the text data
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['string_1'] + df['string_2'])
    
    # Calculate cosine similarity for each pair
    cosine_sim = (tfidf_matrix @ tfidf_matrix.T).A[0]  # Get diagonal as array
    df['cosine_sim'] = cosine_sim

    return df
