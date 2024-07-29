from abc import ABC, abstractmethod
from tqdm import tqdm
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def _load_csv(self):
        return pd.read_csv(self.file_path)
    
    def _load_pickle(self):
        try:
            with open(self.file_path, "rb") as file:
                return pickle.load(file)
        except Exception as e:
            raise ValueError(f"Failed to read pickle file : {str(e)}")
        
    def load_data(self):
        if self.file_path.endswith('.csv'):
            return self._load_csv()
        elif self.file_path.endswith('.pickle') or self.file_path.endswith('.pkl'):
            return self._load_pickle()
        else:
            raise ValueError("Unsupported file type")

class DataService:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    @property
    def dataframe(self):
        return self._dataframe
    
    @dataframe.setter
    def dataframe(self, value):
        if not isinstance(value, pd.DataFrame):
            raise ValueError("Dataset must be a DataFrame object.")
        if "user_id" not in value.columns:
            raise ValueError("Dataset must contains a column named 'user_id'.")
        if "article_id" not in value.columns:
            raise ValueError("Dataset must contains a column named 'article_id'.")
        if "click_timestamp" not in value.columns:
            raise ValueError("Dataset must contains a column named 'click_timestamp'.")
        self._dataframe = value.sort_values(['user_id', 'click_timestamp'], ascending=[True, True])      

    @property
    def unique_users(self):
        return self._dataframe["user_id"].nunique()
    
    @property
    def users_name(self):
        return [int(user) for user in list(self._dataframe["user_id"].unique())]
    
    @property
    def unique_articles(self):
        return self._dataframe["article_id"].nunique()
    
    @property
    def articles_name(self):
        return [int(article) for article in list(self._dataframe["article_id"].unique())]
    
    def __getitem__(self, key):
        return self._dataframe[key]
    
    def get_most_read_articles(self, value):
        if value is None:
            raise TypeError("get_most_read_articles() missing 1 required positional argument: 'value'")
        if not isinstance(value, int) or value <= 0:
            raise ValueError("This method requires a positive integer as parameter.")
        most_read_articles = self._dataframe[['user_id', 'article_id']].groupby('article_id').agg({'user_id': pd.Series.nunique}).sort_values('user_id', ascending=False).head(value)
        return list(most_read_articles.index)
    
class User:
    def __init__(self, id):
        self.id = id

    @property
    def id(self):
        return self._id
    
    @id.setter
    def id(self, value):
        if not value.is_integer():
            raise ValueError("User must be integer.")
        if value < 0:
            raise ValueError("User id must be positive.")
        self._id = value

    def get_read_article(self, data_service, value=0, first=True):
        if not value.is_integer():
            raise ValueError("Value must be integer.")
        if value < 0:
            raise ValueError("Value must be positive.")
        data = data_service.dataframe
        user_article_read = list(data.loc[data['user_id'] == self.id, "article_id"].unique())
        if not value:
            return user_article_read
        if first:
            return user_article_read[:value]
        return user_article_read[-value:]

class ContentBasedModel:
    def __init__(self, embeddings, data_service):
        self.embeddings = embeddings
        self.data_service = data_service
        self.similarities_matrix = cosine_similarity(self.embeddings[self.data_service.articles_name])

    def get_similarity_score(self, idx1, idx2):
        articles = self.data_service.articles_name
        index1 = articles.index(idx1)
        index2 = articles.index(idx2)
        return self.similarities_matrix[index1, index2]
        
    def predict(self, user_id, n_read, n_prediction):
        user_intel = User(user_id)
        articles = self.data_service.articles_name
        user_read_article = user_intel.get_read_article(self.data_service, n_read, first=False)
        matrix_index_read_article = [i for i, j in enumerate(articles) if j in user_read_article]
        user_unread_article = list(set(articles) - set(user_read_article))
        matrix_index_unread_article = [i for i, j in enumerate(articles) if j in user_unread_article]
        articles_similarities = [[unread_article] + [self.similarities_matrix[read_article, unread_article] for read_article in matrix_index_read_article] 
                                 for unread_article in matrix_index_unread_article]
        articles_similarities_sum = [[articles[0], sum(articles[1:])] for articles in articles_similarities]
        sorted_articles = sorted(articles_similarities_sum, key=lambda x: x[1], reverse=True)
        top_articles = sorted_articles[:n_prediction]
        prediction = [int(article[0]) for article in top_articles]
        # Get original article_id.
        prediction = [int(j) for i, j in enumerate(articles) if i in prediction]
        return prediction

class SVD:
    def __init__(self, data_service):
        self.data_service = data_service
        self.user_id_to_index, self.article_id_to_index = self._mapping()

    def _mapping(self):
        user_ids = self.data_service.users_name
        user_id_to_index = {user_column: idx for idx, user_column in enumerate(user_ids)}
        article_ids = self.data_service.articles_name
        article_id_to_index = {article_column: idx for idx, article_column in enumerate(article_ids)}
        return user_id_to_index, article_id_to_index

    def _sparse_matrix(self):
        matrice_base = np.ones(len(self.data_service.dataframe), dtype=int)
        user_column_to_index, article_column_to_index = self._mapping()
        row_indices = self.data_service["user_id"].map(user_column_to_index)
        col_indices = self.data_service["article_id"].map(article_column_to_index)
        ratings_matrix = csr_matrix((matrice_base, (row_indices, col_indices)), shape=(len(self.data_service.users_name), len(self.data_service.articles_name)))
        return ratings_matrix
    
    def fit(self, n_components):
        svd = TruncatedSVD(n_components, random_state=42)
        user_features = svd.fit_transform(self._sparse_matrix())
        item_features = svd.components_
        scores_matrix = user_features.dot(item_features)
        return scores_matrix
    
    def predict(self, user_id, n_articles, scores_matrix):
        user = User(user_id)
        user_index_in_score_matrix = self.user_id_to_index[user.id]
        user_scores = scores_matrix[user_index_in_score_matrix]
        # Normalisation with sigmoïde function.
        user_scores = 1 / (1 + np.exp(-user_scores))
        # Get index of article id for sparse matrix.
        article_read = user.get_read_article(self.data_service)
        article_index_in_score_matrix = [self.article_id_to_index.get(key) for key in article_read]
        user_scores[article_index_in_score_matrix] = -np.inf
        top_article_indices = np.argsort(user_scores)[-n_articles:][::-1]
        # Convert index in sparse matrix to original article_id
        top_articles = [self.data_service.articles_name[idx] for idx in top_article_indices]
        return top_articles

class Metrics:
    def __init__(self, prediction, reality):
        self.prediction = prediction
        self.reality = reality
        
    def soft_accuracy(self):
        if any(prediction in self.reality for prediction in self.prediction):
            return 1
        else:
            return 0

class RecommendationEvaluation:
    def __init__(self, train_data_service, test_data_service):
        self.train_data_service = train_data_service
        self.test_data_service = test_data_service
        self.scores = dict()

    def simple_recommendation_evaluation(self):
        simple_reco = SimpleRecommendation(5)
        predictions = simple_reco.recommend(self.train_data_service)
        for user in self.train_data_service.users_name:
            reality = User(user).get_read_article(self.test_data_service, 5)
            self.scores[user] = Metrics(predictions, reality).soft_accuracy()
        return self.scores
    
    def content_based_evaluation(self, embeddings):
        content_based_model = ContentBasedModel(embeddings, self.train_data_service)
        for user in tqdm(self.train_data_service.users_name):
            predictions = content_based_model.predict(user, 5, 5)
            reality = User(user).get_read_article(self.test_data_service, 5)
            self.scores[user] = Metrics(predictions, reality).soft_accuracy()
        return self.scores

    def svd_evaluation(self, n_components):
        svd = SVD(self.train_data_service)
        score_matrix = svd.fit(n_components)
        for user in self.train_data_service.users_name:
            predictions = svd.predict(user, 5, score_matrix)
            reality = User(user).get_read_article(self.test_data_service, 5)
            self.scores[user] = Metrics(predictions, reality).soft_accuracy()
        return self.scores

class RecommendationStrategy(ABC):
    def __init__(self, n_articles=5):
        self.n_articles = n_articles # Using setter for init.

    @property
    def n_articles(self):
        return self._n_articles

    @n_articles.setter
    def n_articles(self, value):
        if not value.is_integer():
            raise ValueError("Value must be an integer.")
        if value < 1:
            raise ValueError("Number of articles must be at least 1.")      
        self._n_articles = value    

    @abstractmethod
    def recommend(self, data_service):
        raise NotImplementedError("Subclasses must implement this method.")

class SimpleRecommendation(RecommendationStrategy):
    def __init__(self, n_articles=5):
        super().__init__(n_articles)

    def recommend(self, data_service):
        most_read_articles = data_service.get_most_read_articles(5)
        return most_read_articles

class SVDRecommendation(RecommendationStrategy):
    def __init__(self, data_service, user_id, n_articles=5):
        super().__init__(n_articles)
        self.svd_model = SVD(data_service)
        self.score_matrix = self.svd_model.fit(100)
        self.user_id = user_id
        
    def recommend(self):
        return self.svd_model.predict(self.user_id, self.n_articles, self.score_matrix)