import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from data_manager import get_data
from ISO26000 import classify_actions_rse_ISO26000
from ODD import classify_actions_rse_ODD
from impactscore import classify_actions_rse_IMPACTSCORE

def preprocess_text(text):
    # Tokenization
    tokens = nltk.word_tokenize(text.lower())

    # Élimination des stop words
    stop_words = set(stopwords.words('french'))
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Lemmatisation
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]

    return " ".join(lemmatized_tokens)

def rag_pipeline(call_for_proposal):
    data, _ = get_data()

    # Prétraitement du texte
    call_for_proposal = preprocess_text(call_for_proposal)
    companies = [d['nom_courant_denomination'] for d in data]
    actions = [preprocess_text(d.get('action_rse', '')) for d in data]


    # Classification des actions RSE selon les critères ISO 26000
    iso_classification = [classify_actions_rse_ISO26000(d) for d in data]

    # Classification des actions RSE selon les critères ODD
    odd_classification = [classify_actions_rse_ODD(d) for d in data]

    # Classification des actions RSE selon les critères Impact Score
    impact_classification = [classify_actions_rse_IMPACTSCORE(d) for d in data]

    # Vectorisation
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([call_for_proposal] + actions)

    # Calcul du résultat similaire
    similarities = cosine_similarity(vectors[0], vectors[1:])

    # Obtenir le meilleur score
    top_scores = np.argsort(similarities.flatten())[::-1][:10]

    # Sélectionner les entreprises ayant les meilleurs scores
    results = [(companies[i], similarities[0][i], iso_classification[i], odd_classification[i], impact_classification[i]) for i in top_scores]

    return results
