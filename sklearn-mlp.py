from sklearn.neural_network import MLPClassifier

import gzip

import pickle

with gzip.open('./mnist.pkl.gz') as f_gz:

    train_data,valid_data,test_data = pickle.load(f_gz)

clf = MLPClassifier(solver='sgd',activation = 'identity',max_iter = 10,alpha = 1e-5,hidden_layer_sizes = (100,50),random_state = 1,verbose = True)

clf.fit(train_data[0][:10000],train_data[1][:10000])

print clf.predict(test_data[0][:10])

print(clf.score(test_data[0][:100],test_data[1][:100]))

print(clf.predict_proba(test_data[0][:10]))

