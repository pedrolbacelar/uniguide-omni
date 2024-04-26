from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import json
import streamlit as st
stopwords_eng = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]

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
    vectorizer = TfidfVectorizer(stop_words= stopwords_eng)
    student_vector = vectorizer.fit_transform([student_data])
    universities_vectors = vectorizer.transform(universities_data)

    # Calculate the cosine similarity between the student vector and universities vectors
    similarities = cosine_similarity(student_vector, universities_vectors)

    # ---- Correlate the university similarity with the appropriate name in a dictionary
    universities_similarities = {}
    for i in range(len(universities_names)):
        universities_similarities[universities_names[i]] = similarities[0][i]

    return universities_similarities