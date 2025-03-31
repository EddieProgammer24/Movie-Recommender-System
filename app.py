import streamlit as st
import pandas as pd
import pickle
import traceback
from recommender import get_similar_movies_knn, get_top_predictions_svd
from evaluation import get_metric_description


st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommender System")

# Sidebar config
# Load the data
movies = pd.read_table("data/movies.dat", sep="::", engine="python", header=None,
                       names=["Movie ID", "Title", "Genres"], encoding="ISO-8859-1")

ratings = pd.read_table("data/ratings.dat", sep="::", engine="python", header=None,
                        names=["User ID", "Movie ID", "Rating", "Timestamp"], encoding="ISO-8859-1")

# Merge ratings and movies
ratings_with_movies = ratings.merge(movies, on="Movie ID")

# Compute the average rating for each movie
movie_ratings = ratings_with_movies.groupby('Title').agg(
    avg_rating=('Rating', 'mean'),
    rating_count=('Rating', 'count')
).reset_index()

# Filter movies with at least 100 ratings
filtered_movies = movie_ratings[movie_ratings['rating_count'] >= 100]

# Create a slider to select the top k movies
k = st.sidebar.slider('Select number of top movies:', min_value=1, max_value=20, value=10)

# Sort and display top k movies
top_k_movies = filtered_movies.sort_values(by='avg_rating', ascending=False).head(k)

st.write(f"### Top {k} Movies with Highest Average Rating (Min 100 Ratings)")
st.dataframe(top_k_movies)


st.sidebar.markdown("### 🔧 Settings")
algorithm = st.sidebar.selectbox("Choose Algorithm", ["KNN", "SVD"])
metric = st.sidebar.selectbox(
    "Select Evaluation Metric",
    ["RMSE", "Precision@5", "Precision@10", "Diversity", "Novelty", "Coverage"]
)

# Load data
try:
    movies = pd.read_csv("data/movies.dat", sep="::", engine="python", names=["MovieID", "Title", "Genres"], encoding="latin1")
    ratings = pd.read_csv("data/ratings.dat", sep="::", engine="python",  names=["UserID", "MovieID", "Rating", "Timestamp"], encoding="latin1")
    images = pd.read_csv("data/ml1m_images.csv")
    images.rename(columns={"item_id": "MovieID", "image": "ImageURL"}, inplace=True)
    movies = movies.merge(images, on="MovieID", how="left")
    movies.drop_duplicates(subset="MovieID", inplace=True)
except Exception as e:
    st.error("Error loading data.")
    st.text(traceback.format_exc())
    st.stop()

# Load model
model_file = f"models/{algorithm.lower()}_model.pkl"
try:
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    st.sidebar.success(f"{algorithm} model loaded.")
except Exception as e:
    st.sidebar.error(f"{algorithm} model not found.")
    st.text(traceback.format_exc())
    st.stop()

# Movie selection
movie_titles = movies.drop_duplicates("Title")["Title"].sort_values().tolist()
selected_movie = st.selectbox("🎥 Choose a Movie", movie_titles)

# Metric info
st.markdown(f"### 📊 Evaluation Metric: {metric}")
st.info(get_metric_description(metric))

# Generate recommendations
if st.button("🔍 Recommend"):
    try:
        selected_id = movies[movies["Title"] == selected_movie]["MovieID"].values[0]
        st.markdown("## 🎯 Recommended Movies")

        if algorithm == "KNN":
            recs = get_similar_movies_knn(model, selected_id, movies)
        else:
            movie_ids = movies["MovieID"].tolist()
            top_ids = get_top_predictions_svd(model, movie_ids)
            recs = movies[movies["MovieID"].isin(top_ids)]

        for _, row in recs.iterrows():
            title = row["Title"]
            image = row["ImageURL"]

            col1, col2 = st.columns([1, 5])
            with col1:
                if isinstance(image, str) and image.startswith("http"):
                    st.image(image, width=120)
                else:
                    st.image("https://via.placeholder.com/120x180.png?text=No+Image", width=120)
            with col2:
                st.markdown(f"{title}")
                st.markdown(f"⭐ Algorithm: {algorithm}")
                st.markdown("---")
    except Exception as e:
        st.error("❌ Error generating recommendations.")
        st.text(traceback.format_exc())

# Show popular movies
st.markdown("## 🍿 Popular Movies")
try:
    top_ids = ratings["MovieID"].value_counts().head(12).index
    popular_movies = movies[movies["MovieID"].isin(top_ids)]

    cols = st.columns(4)
    for i, (_, row) in enumerate(popular_movies.iterrows()):
        with cols[i % 4]:
            if isinstance(row["ImageURL"], str) and row["ImageURL"].startswith("http"):
                st.image(row["ImageURL"], use_container_width=True)
            else:
                st.image("https://via.placeholder.com/120x180.png?text=No+Image", use_container_width=True)
            st.caption(row["Title"])
except Exception as e:
    st.error("Error loading popular movies.")
    st.text(traceback.format_exc())