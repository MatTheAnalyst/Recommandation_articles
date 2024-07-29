from utils import DataLoader, DataService, RecommendationEvaluation

# Load data.
train_set_loader = DataLoader("prepare_data/train.csv")
test_set_loader = DataLoader("prepare_data/test.csv")
train_dataservice = DataService(train_set_loader.load_data())
test_dataservice = DataService(test_set_loader.load_data())

recommendation = RecommendationEvaluation(train_dataservice, test_dataservice)
scores = recommendation.svd_evaluation(100)
print(f"SVD model scores : {sum(scores.values())} ({sum(scores.values())*100/len(scores):.2f} %)")