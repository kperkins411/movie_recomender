#!/usr/bin/env python

# USAGE
# python parse_data.py

#splits the train data into 90% train and 10% validation
import logging
import sys
import numpy as np

# data = np.random.random((1000, 100))
# labels = np.random.randint(10, size=(1000, 1))


sys.path.append('../KP_utils')
import utils

logging.basicConfig(
    #filename = 'parse_data.log', #comment this line out if you want data in the console
    format = "%(levelname) -10s %(asctime)s %(module)s:%(lineno)s %(funcName)s %(message)s",
    level = logging.DEBUG
)

import pandas as pd
import os
from keras.layers import Input, Embedding
from keras.regularizers import l2

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.regularizers import l2, l1

from keras.optimizers import SGD, RMSprop, Adam

path = "data/"
model_path = path + 'models/'
if not os.path.exists(model_path): os.mkdir(model_path)
batch_size=64

ratings = pd.read_csv(path+'ratings.csv')
ratings.head()

movie_names = pd.read_csv(path+'movies.csv').set_index('movieId')['title'].to_dict()

# users = sorted(ratings.userId.unique())
# movies = sorted(ratings.movieId.unique())

users = ratings.userId.unique()
movies = ratings.movieId.unique()

userid2idx = {o:i for i,o in enumerate(users)}
movieid2idx = {o:i for i,o in enumerate(movies)}

ratings.movieId = ratings.movieId.apply(lambda x: movieid2idx[x])
ratings.userId = ratings.userId.apply(lambda x: userid2idx[x])

user_min, user_max, movie_min, movie_max = (ratings.userId.min(),
    ratings.userId.max(), ratings.movieId.min(), ratings.movieId.max())

n_users = ratings.userId.nunique()
n_movies = ratings.movieId.nunique()
n_users, n_movies

# This is the number of latent factors in each embedding.
n_factors = 50
np.random.seed = 42

msk = np.random.rand(len(ratings)) < 0.8
trn = ratings[msk]
val = ratings[~msk]


def embedding_input(name, n_in, n_out, reg):
    inp = Input(shape=(1,), dtype='int64', name=name)
    return inp, Embedding(n_in, n_out, input_length=1, embeddings_regularizer=l2(reg))(inp)
user_in, u = embedding_input('user_in', n_users, n_factors, 1e-4)
movie_in, m = embedding_input('movie_in', n_movies, n_factors, 1e-4)


# create subset
g=ratings.groupby('userId')['rating'].count()
topUsers=g.sort_values(ascending=False)[:15]

g=ratings.groupby('movieId')['rating'].count()
topMovies=g.sort_values(ascending=False)[:15]

top_r = ratings.join(topUsers, rsuffix='_r', how='inner', on='userId')
top_r = top_r.join(topMovies, rsuffix='_r', how='inner', on='movieId')
pd.crosstab(top_r.userId, top_r.movieId, top_r.rating, aggfunc=np.sum)

# Neural net
# Rather than creating a special purpose architecture (like our dot-product with bias earlier), it's often both easier
# and more accurate to use a standard neural network. Let's try it! Here, we simply concatenate the user and movie
# embeddings into a single vector, which we feed into the neural net.

user_in, u = embedding_input('user_in', n_users, n_factors, 1e-4)
movie_in, m = embedding_input('movie_in', n_movies, n_factors, 1e-4)

x = merge([u, m], mode='concat')
x = Flatten()(x)
x = Dropout(0.3)(x)
x = Dense(70, activation='relu')(x)
x = Dropout(0.75)(x)
x = Dense(1)(x)
nn = Model([user_in, movie_in], x)
nn.compile(Adam(0.001), loss='mse')

nn.fit([trn.userId, trn.movieId], trn.rating, batch_size=64, nb_epoch=8,
          validation_data=([val.userId, val.movieId], val.rating))
pass


