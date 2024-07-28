import azure.functions as func
import os
import json
import logging
import pandas as pd
import pickle
import h5py
import numpy as np
import pickle

DATA_NAME = "articles_clients.csv"
MODEL_NAME = "model_recommendation.pkl"
USER_FEATURE_NAME = "user_features.h5"

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

def read_pickle_file(file_path:str):
    try:
        with open(file_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        logging.error(f"Failed to read pickle blob : {str(e)}")
        return None

def read_h5_file(file_path:str):
    try:
        with h5py.File(file_path, 'r') as hf:
            h5_file = hf['user_features_100'][:]
        return h5_file
    except Exception as e:
        logging.error(f"Failed to read h5 : {str(e)}")
        return None

def read_csv_file(file_path:str):
    try:
        csv_file = pd.read_csv(file_path, encoding="utf-8")
        return csv_file
    except Exception as e:
        logging.error(f"Failed to read file : {str(e)}")
        return None

@app.function_name('FirstHTTPFunction')
@app.route(route="welcome", auth_level=func.AuthLevel.ANONYMOUS)
def function_for_testing(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    try:
        param = req.params.get('param', 'default')
        embedding = read_pickle_file(MODEL_NAME)
        if len(embedding.explained_variance_) > 0:
            message = f"Here is the csv for {param}: {embedding.explained_variance_}"
            return func.HttpResponse(message, status_code=200)
        else:
            return func.HttpResponse("Failed to load CSV.", status_code=500)
    except Exception as e:
        return func.HttpResponse(f"An error occurred: {str(e)}", status_code=500)

@app.function_name('SecondHTTPFunction')
@app.route(route="newroute", auth_level=func.AuthLevel.ANONYMOUS)
def test_function(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Starting the second HTTP Function request.')
    user_id = req.params.get('user_id')
    if user_id:
        message = f"Hello, {user_id}, so glad this Function worked!!"
    else:
        message = "Hello, so glad this Function worked!!"
    return func.HttpResponse(
        message,
        status_code=200
    )

def mapping(data, user_column:str = 'user_id', article_column:str = 'article_id'):
    user_ids = list(data[user_column].unique())
    user_id_to_index = {user_column: idx for idx, user_column in enumerate(user_ids)}

    article_ids = list(data[article_column].unique())
    article_id_to_index = {article_column: idx for idx, article_column in enumerate(article_ids)}

    return user_id_to_index, article_id_to_index

def get_article_read(data:pd.DataFrame, user_id:int, user_column:str = 'user_id', article_column:str = 'article_id', number_of_article:int = 0, first:bool = True):
    if user_column not in data.columns:
        raise ValueError(f"{user_column} not in dataframe provided")
    elif article_column not in data.columns:
        raise ValueError(f"{article_column} not in dataframe provided")
    
    user_ids = list(data[user_column].unique())
    if user_id not in user_ids:
        raise ValueError(f"{user_id} not in user in dataframe provided")
    
    if number_of_article < 0:
        raise ValueError(f"{number_of_article} must be >= 0")
    
    user_article_read = list(data.loc[data['user_id'] == user_id, article_column].unique())

    if first:
        if number_of_article == 0 or number_of_article >= len(user_article_read):
            return user_article_read
        else:
            return user_article_read[:number_of_article]
    else:
        if number_of_article == 0 or number_of_article >= len(user_article_read):
            return user_article_read
        else:
            return user_article_read[-number_of_article:]

def predict_top_n_articles(data, user_id, svd, user_features, n:int = 5, article_column:str = 'article_id'):
    user_id_to_index, article_id_to_index = mapping(data=data)
    article_ids = list(data[article_column].unique())
    user_index = user_id_to_index[user_id]
    read_articles = get_article_read(data=data, user_id=user_id)

    # Convertir les IDs d'articles lus en indices
    read_article_indices = [article_id_to_index.get(article_id, None) for article_id in read_articles if article_id in article_id_to_index]
    
    # Calculer les scores pour tous les articles, en excluant ceux déjà lus
    scores = np.dot(user_features[user_index], svd.components_)
    # Normaliser les scores avec une fonction sigmoïde
    scores = 1 / (1 + np.exp(-scores))
    scores[read_article_indices] = -np.inf  # Exclusion des articles lus
    
    # Obtenir les indices des articles avec les plus hauts scores
    top_article_indices = np.argsort(scores)[-n:][::-1]
    
    # Convertir les indices en article_id
    top_articles = [article_ids[idx] for idx in top_article_indices]
    top_articles = [int(a) for a in top_articles]
    
    return top_articles

@app.function_name('RecommendationHTTPFunction')
@app.route(route="predict", auth_level=func.AuthLevel.ANONYMOUS)
def prediction_function(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Starting the recommendation Function request.')
    user_id = req.params.get('user_id')
    if user_id:
        articles_clients = read_csv_file(DATA_NAME)
        model = read_pickle_file(MODEL_NAME)
        user_features = read_h5_file(USER_FEATURE_NAME)

        prediction = predict_top_n_articles(data=articles_clients, user_id=int(user_id), svd=model, user_features=user_features)
        message = f"Here is the prediction list : {prediction}"
    else:
        message = "No user_id provided, please provide user_id. End point example : ?user_id=1"
    return func.HttpResponse(
        message,
        status_code=200
    )