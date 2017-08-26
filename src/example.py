import pickle

with open('result/task8_train.pickle', 'rb') as fd:
    train_data, train_target = pickle.load(fd)

with open('result/task8_test.pickle', 'rb') as fd:
    test_data, test_target = pickle.load(fd)
