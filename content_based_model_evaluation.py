from utils import DataLoader, DataService, RecommendationEvaluation

# Load data.
train_set_loader = DataLoader("prepare_data/train.csv")
test_set_loader = DataLoader("prepare_data/test.csv")
train_dataservice = DataService(train_set_loader.load_data())
test_dataservice = DataService(test_set_loader.load_data())
embeddings_loader = DataLoader("prepare_data/articles_embeddings.pickle")
embeddings = embeddings_loader.load_data()

recommendation = RecommendationEvaluation(train_dataservice, test_dataservice)
recommendation.content_based_evaluation(embeddings)