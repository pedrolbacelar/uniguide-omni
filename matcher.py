from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import streamlit as st


def load_student_data():
    #--- load the json file
    with open("cache-data.json") as f:
        data = json.load(f)
    return data["user_profile"]

def load_universities_database():
    loaded_universities_data = []
    loaded_universities_names = []

    #--- load the json file
    with open("universities_database.json") as f:
        data = json.load(f)

    loaded_universities_names = list(data.keys())
    loaded_universities_data = list(data.values())

    return loaded_universities_data, loaded_universities_names
    
    

def match(student_data, universities_data, universities_names):
    # Vectorize the student data and universities data
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
    student_vector = vectorizer.fit_transform([student_data])
    universities_vectors = vectorizer.transform(universities_data)

    # Calculate the cosine similarity between the student vector and universities vectors
    similarities = cosine_similarity(student_vector, universities_vectors)

    # ---- Correlate the university similarity with the appropriate name in a dictionary
    universities_similarities = {}
    for i in range(len(universities_names)):
        universities_similarities[universities_names[i]] = similarities[0][i]

    return universities_similarities