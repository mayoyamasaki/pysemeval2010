import pickle

from embeddings import Embeddings

num_features = 50
WE = Embeddings('data/glove.6B/sampled-glove.6B.50d.txt', num_features)

with open('result/task8_train.pickle', 'rb') as fd:
    train_data, train_target = pickle.load(fd)

with open('result/task8_test.pickle', 'rb') as fd:
    test_data, test_target = pickle.load(fd)
