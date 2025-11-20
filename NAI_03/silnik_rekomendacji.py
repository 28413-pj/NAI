"""
==============================================================
Movie Recommendation System Using K-Means Clustering
==============================================================

 Problem Description
----------------------
    This script loads a dataset of movie ratings, groups users by similarity
    using K-Means clustering, and recommends movies to a selected user based
    on what similar users enjoyed. It also provides a list of movies that
    similar users rated poorly (anti-recommendations). The script further
    queries the OMDB API to fetch metadata for each suggested movie.

 Environment Setup
---------------------
    1. Ensure that "Oceny.csv" exists in the same directory as this script.
        The file must contain the columns: user, title, rating.
    2. Install required dependencies
        - pandas
        - scikit-learn
        - requests
       Example installation:
        pip install pandas scikit-learn requests
    3. Run the script with:
       python silnik_rekomendacji.py
    4. The script will output recommendations, anti-recommendations, and
       metadata fetched from the OMDB API.

 Authors
------------
• Marek Jenczyk
• Oskar Skomra

"""

import pandas as pd
from sklearn.cluster import KMeans
import requests

# --- Load data ---
df = pd.read_csv("Oceny.csv", sep=";", encoding="cp1250")

# User × Movie matrix
matrix = df.pivot_table(index="user", columns="title", values="rating")

# Fill missing ratings with the user's mean rating
matrix_filled = matrix.apply(lambda r: r.fillna(r.mean()), axis=1)

# --- K-Means clustering ---
kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
labels = kmeans.fit_predict(matrix_filled)

matrix_filled["cluster"] = labels

# --- Select target user ---
target_user = "Oskar Skomra"
user_cluster = matrix_filled.loc[target_user, "cluster"]

# Users from the same cluster
cluster_users = matrix_filled[matrix_filled["cluster"] == user_cluster].drop(columns="cluster")

# Average ratings within the cluster
cluster_means = cluster_users.mean()

# Movies not rated by the target user
unwatched = matrix.loc[target_user][matrix.loc[target_user].isna()].index

# Recommendations and anti-recommendations
recommendations = cluster_means[unwatched].sort_values(ascending=False).head(5)
antirec = cluster_means[unwatched].sort_values().head(5)


def get_movie_info(title):
    """
    Fetch metadata about a movie from the OMDB API.

    Parameters:
        title (str): The movie title.

    Returns:
        dict: Information including Year, Genre, Runtime, and IMDb rating.
              Returns an error message if the API responds unsuccessfully.
    """
    url = f"https://www.omdbapi.com/?t={title}&apikey=8e678668"
    response = requests.get(url)

    if response.status_code != 200:
        return {"Error": f"HTTP {response.status_code}"}

    data = response.json()

    if data.get("Response") == "False":
        return {"Error": data.get("Error")}

    return {
        "Year": data.get("Year"),
        "Genre": data.get("Genre"),
        "Runtime": data.get("Runtime"),
        "IMDb Rating": data.get("imdbRating"),
    }


def print_movies(title, series):
    """
    Print a list of movies with engine ratings and metadata.

    Parameters:
        title (str): Title of the printed section (e.g., recommendations).
        series (pd.Series): Movies mapped to their predicted rating.
    """
    print(f"\n=== {title} ===")
    for film in series.index:
        engine_rating = series.loc[film]
        info = get_movie_info(film)

        print(f"\n--- {film} ---")
        print(f"Engine rating: {engine_rating:.2f}")

        if "Error" in info:
            print(f"API Error: {info['Error']}")
            continue

        print(f"Year: {info['Year']}")
        print(f"Genre: {info['Genre']}")
        print(f"Runtime: {info['Runtime']}")
        print(f"IMDb Rating: {info['IMDb Rating']}")


# --- Output results ---
print("\nUser:", target_user)

print_movies("Top 5 recommendations (with movie info)", recommendations)
print_movies("Top 5 anti-recommendations (with movie info)", antirec)
