def get_metric_description(metric_name):
    descriptions = {
        "RMSE": "Root Mean Squared Error — how far off predictions are on average.",
        "Precision@5": "Proportion of top-5 results that are relevant.",
        "Precision@10": "Proportion of top-10 results that are relevant.",
        "Diversity": "How different the recommended items are from each other.",
        "Novelty": "How uncommon the recommended items are.",
        "Coverage": "Percentage of the catalog recommended across users."
    }
    return descriptions.get(metric_name, "No description available.")