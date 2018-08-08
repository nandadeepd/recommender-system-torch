import sys, torch, numpy as np
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.factorization.explicit import ExplicitFactorizationModel
from spotlight.cross_validation import random_train_test_split
from spotlight.evaluation import rmse_score

# testing with 100k user interactions 
dataset = get_movielens_dataset(variant='100K')

# base, low rank factorization adopted in netflix
# part of collaborative filtering algorithm that uses implicit and explicit interaction matrices. 
# spotlight gives out the interactions class for each dataset curated. 
# Hyperparameters:
# loss function
# epochs
model = ExplicitFactorizationModel(loss='regression', embedding_dim=256, n_iter=50, batch_size=1024, l2=1e-9, learning_rate=1e-3, use_cuda=torch.cuda.is_available())

# split training and testing data
train, test = random_train_test_split(dataset, random_state=np.random.RandomState(42))
# print('Split into \n {} and \n {}.'.format(train, test))

# fit the model - likely to overfit
model.fit(train, verbose=True)

# using root mean squared error measure
train_rmse = rmse_score(model, train)
test_rmse = rmse_score(model, test)
print('Train error {:.2f}, test error {:.2f}'.format(train_rmse, test_rmse))

