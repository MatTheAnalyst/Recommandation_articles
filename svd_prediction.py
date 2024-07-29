from utils import DataLoader, DataService, RecommendationStrategy, SVD

class SVDRecommendation(RecommendationStrategy):
    def __init__(self, data_service, user_id, n_articles=5):
        super().__init__(n_articles)
        self.svd_model = SVD(data_service)
        self.score_matrix = self.svd_model.fit(100)
        self.user_id = user_id
    def recommend(self):
        return self.svd_model.predict(self.user_id, self.n_articles, self.score_matrix)

USER_ID = 1

# Load data.
train_set_loader = DataLoader("prepare_data/train.csv")
train_dataservice = DataService(train_set_loader.load_data())

svd_model = SVDRecommendation(train_dataservice, USER_ID)

print(f"SVD model recommendations : {svd_model.recommend()}")