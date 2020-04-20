import pickle
import pandas as pd
import numpy as np
import random as rd
import keras
from keras.optimizers import SGD
from keras.regularizers import l1
from tensorflow.keras.optimizers import SGD
from keras import backend as K
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, StandardScaler
from tensorflow_core.python.keras import Input
from tensorflow_core.python.keras.layers import Dense, Activation
from tensorflow_core.python.keras.models import Model, Sequential
from tensorflow_core.python.keras.utils.vis_utils import plot_model

df = pd.read_csv('u.data', sep='\t', header=None, usecols=[0, 1, 2])
df = df.head(10)
df.columns = ['userid', 'movie', 'rating']

unique_user = df['userid'].unique()  # all unique users
unique_movies = df['movie'].unique()  # all unique movies

# initialization

mean_by_id = {}
X = []
Y = []

# # CALCULATING THE MEAN
#
# user_mean = df.groupby(['userid'])
# for id in df['userid']:
#     mean_by_id[id] = user_mean.get_group(id)['rating'].mean()
#     # get all the ratings of the user with the same id, and calculate the mean
#
# # SUBTRACTING THE MEAN
#
# depth = int(df.size / 3)
# df_centered = list()
# for item in range(depth):
#     row = df.iloc[item]
#     df_centered.append(row['rating'] - mean_by_id[row['userid']])  # subtract the mean
# df_centered = pd.DataFrame(df_centered)  # convert centered values to a Dataframe
# df['rating'] = df_centered  # add the dataframe to the "rating" column of the first dataframe
#
# # ADDING MISSING VALUES
#
# df.columns = [0, 1, 2]  # renaming the columns for easier indexing
# df_not_rated = []
# for userid in unique_user:  # loop over the unique users
#     for movie in unique_movies:  # check for all the movies
#         if not df[(df[0] == userid) & (df[1] == movie)].any().any():
#             # new_rate = rd.randrange(0, 6) # get a random value to fill the empty values
#             new_rate = mean_by_id[userid]  # get the mean of the user
#             df_not_rated.append([userid, movie, new_rate])  # filling the non rated movies with random number
# df = df.append(df_not_rated)  # df + df_not_rated
# df = df.reset_index(drop=True)  # resetting the indexes after appending
#
# df.columns = ['userid', 'movie', 'rating']
#
# # RESCALING THE VALUES TO 0 -> 1
#
# scaler = MinMaxScaler()  # rescaler object
# df['rating'] = scaler.fit_transform(df['rating'].values.reshape(-1, 1))  # default rescale is from 0 to 1

# # pickle save # #

# df.to_pickle('collab_filtering.pkl')

# # read from pickle # #

infile = open('collab_filtering.pkl', 'rb')
df = pickle.load(infile)
infile.close
unique_user = df['userid'].unique()
unique_movies = df['movie'].unique()

# ONE HOT ENCODING
ohe = OneHotEncoder()  # initialize an OneHotEncoder object
userids = unique_user.reshape(-1, 1)
one_hot_encoded = ohe.fit_transform(userids)  # calculate the one hot enc.

_x = zip(unique_user, one_hot_encoded)  # concat ids and hot encodings
one_hot_encoded = dict(_x)  # add it to a dict for {id: one hot enc} representation

# INPUT X

for index in unique_user:
    _temp = one_hot_encoded[index]  # get from dict the one hot
    X.append(_temp.toarray().squeeze())  # add one hot encodings to the INPUT X

# OUTPUT Y

for id in unique_user:
    movies = df[df['userid'] == id]  # get all the movies from the user
    movies = movies.sort_values(by='movie')  # sort them by movie value
    Y.append(movies['rating'].values)  # append them to the final output

X = np.asarray(X)
Y = np.asarray(Y)

df = StandardScaler().fit_transform(X=df)

kfold = KFold(n_splits=5, shuffle=True)
rmseList = []
rrseList = []


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


opt = SGD(lr=0.001, momentum=0.6, nesterov=False)
print("begin training")
for i, (train, test) in enumerate(kfold.split(X)):
    model_in = Input(shape=(len(unique_user, )))
    x = Dense(10, activation='relu')(model_in)
    model_out1 = Dense(10, activation='relu')(model_in)
    model_out2 = Dense(10, activation='relu')(model_in)
    model_out = Dense(len(unique_movies), activation='sigmoid')(model_out2)
    model = Model(inputs=model_in, outputs=model_out)
    model.compile(loss='mae', optimizer=opt, metrics=['mae', 'acc'])
    history = model.fit(X[train], Y[train], validation_data=(X[test], Y[test]), epochs=400, batch_size=10, verbose=0)

    scores = model.evaluate(X[test], Y[test])
    rmseList.append(scores[1])
    print("Fold :", i, " RMSE:", scores[1])
# pyplot.plot(rmseList)
# pyplot.show()

print(np.mean(rmseList))
print(history.history.keys())

train_mse = model.evaluate(X[train], Y[train], verbose=0)
test_mse = model.evaluate(X[test], Y[test], verbose=0)
# pyplot.title('Loss')
# pyplot.plot(history.history['loss'], label='train')
# pyplot.plot(history.history['val_loss'], label='test')

# pyplot.legend()
# pyplot.show()
