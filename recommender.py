import pandas as pd

def get_similar_movies_knn(model, movie_id, movies_df, k=5):
    try:
        inner_id = model.trainset.to_inner_iid(movie_id)
        neighbor_ids = model.get_neighbors(inner_id, k=k)
        recommended_ids = [int(model.trainset.to_raw_iid(inner_id)) for inner_id in neighbor_ids]
        return movies_df[movies_df["MovieID"].isin(recommended_ids)]
    except Exception:
        return pd.DataFrame()

def get_top_predictions_svd(model, movie_id_list, user_id=0, k=5):
    predictions = [model.predict(user_id, movie_id) for movie_id in movie_id_list]
    sorted_preds = sorted(predictions, key=lambda x: x.est, reverse=True)[:k]
    top_ids = [int(pred.iid) for pred in sorted_preds]
    return top_ids