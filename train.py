import pandas as pd
import pickle
import os
from surprise import Dataset, Reader, KNNBasic, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

import os

# Load data
ratings = pd.read_csv("data/ratings.dat", delimiter="::", engine="python", 
                      names=["UserID", "MovieID", "Rating", "Timestamp"])

# Define rating scale and load data
reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[["UserID", "MovieID", "Rating"]], reader)
# Split into train and test sets (80-20)
trainset, testset = train_test_split(data, test_size=0.2)

print(ratings.head())
print(ratings.columns)

# Train KNN model
knn_model = KNNBasic(sim_options={"name": "cosine", "user_based": False})
knn_model.fit(trainset)

# Train SVD model
svd_model = SVD()
svd_model.fit(trainset)

# Evaluate KNN model
knn_predictions = knn_model.test(testset)
knn_rmse = accuracy.rmse(knn_predictions)
print(f"KNN Model RMSE: {knn_rmse:.4f}")

# Evaluate SVD model
svd_predictions = svd_model.test(testset)
svd_rmse = accuracy.rmse(svd_predictions)
print(f"SVD Model RMSE: {svd_rmse:.4f}")

# Save both models
os.makedirs("models", exist_ok=True)
with open("models/knn_model.pkl", "wb") as f:
    pickle.dump(knn_model, f)

with open("models/svd_model.pkl", "wb") as f:
    pickle.dump(svd_model, f)

print("✅ Models trained and saved.")