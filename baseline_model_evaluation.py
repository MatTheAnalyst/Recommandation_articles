from utils import DataLoader, DataService, RecommendationStrategy, RecommendationEvaluation

class SimpleRecommendation(RecommendationStrategy):
    def __init__(self, n_articles=5):
        super().__init__(n_articles)

    def recommend(self, data_service):
        most_read_articles = data_service.get_most_read_articles(5)
        return most_read_articles

# Load data.
train_set_loader = DataLoader("prepare_data/train.csv")
test_set_loader = DataLoader("prepare_data/test.csv")
train_dataservice = DataService(train_set_loader.load_data())
test_dataservice = DataService(test_set_loader.load_data())

# Baseline model.
prediction = SimpleRecommendation(5)
scores = RecommendationEvaluation(train_dataservice, test_dataservice).simple_recommendation_evaluation()
print(f"Baseline model scores : {sum(scores.values())} ({sum(scores.values())*100/len(scores):.2f} %)")
