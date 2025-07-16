import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import os
import ast
import time
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =================== TMDB API KEY ===================
API_KEY = "aae137997fefedd4a27940246374521d"  # Replace with your TMDB key

# =================== Retry-safe GET function ===================
def safe_get(url, retries=3, delay=2):
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response
        except requests.exceptions.ConnectTimeout:
            print(f"[Timeout] Attempt {attempt+1} failed: {url}")
        except requests.exceptions.RequestException as e:
            print(f"[Error] Attempt {attempt+1}: {e}")
        time.sleep(delay)
    return None

# =================== Fetch Poster and Details ===================
@st.cache_data(show_spinner=False)
def fetch_movie_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    response = safe_get(url)
    if response is None:
        return {
            "poster": None,
            "title": "Error",
            "genres": "N/A",
            "rating": "N/A",
            "overview": "⚠️ Failed to fetch details (timeout)",
            "release_date": "N/A",
            "id": movie_id
        }
    data = response.json()
    return {
        "poster": "https://image.tmdb.org/t/p/w500/" + data.get('poster_path') if data.get('poster_path') else None,
        "title": data.get("title"),
        "genres": ", ".join([genre['name'] for genre in data.get("genres", [])]),
        "rating": data.get("vote_average"),
        "overview": data.get("overview"),
        "release_date": data.get("release_date"),
        "id": movie_id
    }

# =================== Fetch Trailer Link ===================
@st.cache_data(show_spinner=False)
def fetch_movie_trailer(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}&language=en-US"
    response = safe_get(url)
    if response is None:
        return None
    data = response.json()
    for video in data.get("results", []):
        if video["site"] == "YouTube" and video["type"] == "Trailer":
            return f"https://www.youtube.com/watch?v={video['key']}"
    return None

# =================== Data Processing Functions ===================
def convert(obj):
    return [i['name'] for i in ast.literal_eval(obj)]

def convert3(obj):
    return [i['name'] for i in ast.literal_eval(obj)[:3]]

def fetch_director(obj):
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            return [i['name']]
    return []

def stem(text):
    ps = PorterStemmer()
    return " ".join([ps.stem(i) for i in text.split()])

# =================== Generate Pickle Files ===================
def generate_data():
    movies = pd.read_csv("tmdb_5000_movies.csv")
    credits = pd.read_csv("tmdb_5000_credits.csv")
    movies = movies.merge(credits, on='title')
    movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
    movies.dropna(inplace=True)

    movies['genres'] = movies['genres'].apply(convert)
    movies['keywords'] = movies['keywords'].apply(convert)
    movies['cast'] = movies['cast'].apply(convert3)
    movies['crew'] = movies['crew'].apply(fetch_director)
    movies['overview'] = movies['overview'].apply(lambda x: x.split())

    for feature in ['genres', 'keywords', 'cast', 'crew']:
        movies[feature] = movies[feature].apply(lambda x: [i.replace(" ", "") for i in x])

    movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
    new_df = movies[['movie_id', 'title', 'tags']]
    new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
    new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
    new_df['tags'] = new_df['tags'].apply(stem)

    cv = CountVectorizer(max_features=5000, stop_words='english')
    vectors = cv.fit_transform(new_df['tags']).toarray()

    similarity = cosine_similarity(vectors)

    pickle.dump(new_df, open('movies.pkl', 'wb'))
    pickle.dump(similarity, open('similarity.pkl', 'wb'))

# =================== Recommend Function ===================
def recommend(movie):
    movie_index = movies_df[movies_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended = []
    for i in movie_list:
        movie_id = movies_df.iloc[i[0]].movie_id
        details = fetch_movie_details(movie_id)
        trailer_url = fetch_movie_trailer(movie_id)
        details['trailer_url'] = trailer_url
        recommended.append(details)
    return recommended

# =================== Main App ===================
if not os.path.exists("movies.pkl") or not os.path.exists("similarity.pkl"):
    generate_data()

movies_df = pickle.load(open("movies.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

st.title("🎬 Movie Recommender System")

selected_movie = st.selectbox("Select a movie to get recommendations", movies_df['title'].values)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    for idx, movie in enumerate(recommendations):
        st.subheader(movie["title"])
        cols = st.columns([1, 2])
        with cols[0]:
            if movie["poster"]:
                st.image(movie["poster"])
            else:
                st.text("No Image")
        with cols[1]:
            st.markdown(f"**Genres:** {movie['genres']}")
            st.markdown(f"**Rating:** {movie['rating']}")
            st.markdown(f"**Release Date:** {movie['release_date']}")
            st.markdown(f"**Overview:** {movie['overview']}")

            if movie['trailer_url']:
                trailer_button = f"""
                    <a href="{movie['trailer_url']}" target="_blank">
                        <button style="background-color:#ff4b4b;color:white;border:none;padding:8px 16px;border-radius:5px;cursor:pointer;">
                            ▶️ Watch Trailer
                        </button>
                    </a>
                """
                st.markdown(trailer_button, unsafe_allow_html=True)
            else:
                st.markdown("🚫 Trailer not available")
        st.markdown("---")
