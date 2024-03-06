#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load movie datasets
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge datasets on 'title'
movies = movies.merge(credits, on='title')

# Select relevant columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Drop rows with missing values
movies.dropna(inplace=True)

# Convert JSON strings in 'genres', 'keywords', 'cast', 'crew' columns to lists
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)][:3])
movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'])

# Tokenize and preprocess 'overview' column
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Collapse lists to strings with removed spaces
def collapse(L):
    return [i.replace(" ","") for i in L]

movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

# Combine relevant features into 'tags' column
movies['tags'] = movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Create a new DataFrame with selected columns
new_df = movies.drop(columns=['overview','genres','keywords','cast','crew'])

# Convert 'tags' column to lowercase and join elements
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Feature extraction using CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()

# Calculate cosine similarity
similarity = cosine_similarity(vector)

# Function to recommend movies based on similarity
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    print("Top 5 recommended movies for {}: ".format(movie))
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)

recommend('Avatar')
# Save data and models using pickle
pickle.dump(new_df, open('movie_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
