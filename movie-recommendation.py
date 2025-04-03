import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Load MovieLens dataset (ensure these files are in the same directory)
try:
    movies = pd.read_csv("movies.csv")
    ratings = pd.read_csv("ratings.csv")
except FileNotFoundError:
    st.error("Error: movies.csv or ratings.csv not found. Please upload them.")
    st.stop()

movies = movies[['movieId', 'title', 'genres']]
ratings = ratings[['userId', 'movieId', 'rating']]

movies.dropna(inplace=True)
ratings.dropna(inplace=True)

tfidf = TfidfVectorizer(stop_words="english")
movies["genres"] = movies["genres"].fillna("")
tfidf_matrix = tfidf.fit_transform(movies["genres"])

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def content_recommendations(movie_title, num_recommendations=5):
    try:
        idx = movies[movies["title"] == movie_title].index[0]
    except IndexError:
        return pd.DataFrame(), []
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num_recommendations+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies.iloc[movie_indices][["movieId", "title", "genres"]], [i[1] for i in sim_scores]

reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2)

svd = SVD()
svd.fit(trainset)

def collaborative_recommendations(user_id, num_recommendations=5):
    if user_id not in ratings['userId'].values:
        return []
    movie_ids = movies['movieId'].unique()
    predictions = [(movie, svd.predict(user_id, movie).est) for movie in movie_ids]
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:num_recommendations]
    return predictions

def hybrid_recommendations(user_id, movie_title, weight_content=0.4, weight_collab=0.6, num_recommendations=5):
    content_based, content_scores = content_recommendations(movie_title, num_recommendations)
    collaborative_based = collaborative_recommendations(user_id, num_recommendations)

    hybrid_scores = {}

    if not content_based.empty:
        for index, row in content_based.iterrows():
            if index < len(content_scores):
                hybrid_scores[row["movieId"]] = {"title": row["title"], "score": weight_content * content_scores[index]}

    if collaborative_based:
        for movie_id, prediction_score in collaborative_based:
            if movie_id in hybrid_scores:
                hybrid_scores[movie_id]["score"] += weight_collab * prediction_score
            else:
                hybrid_scores[movie_id] = {"title": movies[movies['movieId'] == movie_id]["title"].values[0], "score": weight_collab * prediction_score}

    sorted_movies = sorted(hybrid_scores.items(), key=lambda x: x[1]["score"], reverse=True)

    return [(movies[movies['movieId'] == movie_id]["title"].values[0], score["score"]) for movie_id, score in sorted_movies]

st.title("Movie Recommendation System")

user_id = st.number_input("Enter User ID:", min_value=1, value=1)
movie_title = st.selectbox("Select Movie Title:", movies["title"].unique())

if st.button("Get Recommendations"):
    recommendations = hybrid_recommendations(user_id, movie_title, num_recommendations=5)

    if recommendations:
        st.subheader(f"Top Hybrid Recommendations for User {user_id} based on '{movie_title}':")
        for idx, (movie, score) in enumerate(recommendations, start=1):
            st.write(f"{idx}. {movie} (Score: {score:.2f})")
    else:
        st.write("No recommendations found for this user.")