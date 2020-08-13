import os

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import adam, RMSprop
from keras.regularizers import l2, l1

from graph_convolution import GraphConv

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import tensorflow as tf

from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns

def r_square_np(y_true, y_pred):
    '''
    Calcualte the R^2 coefficent between y_true and y_pred
    '''
    y_pred_mean = np.mean(y_pred)
    y_true_mean = np.mean(y_true)
    square_corr_values = np.square((np.sum((y_pred-y_pred_mean)*(y_true-y_true_mean)))/(np.sqrt(np.sum(np.square(y_pred-y_pred_mean))*np.sum(np.square(y_true-y_true_mean)))))
    return square_corr_values


if __name__ == '__main__':

    batch_size = 200
    epochs = 40
    num_neighbors = 5
    filters_1 = 10
    filters_2 = 20
    num_hidden_1 = 300
    num_hidden_2 = 100
    results = dict()

    tr_loc = '../data/DPP4_training_disguised.csv'
    val_loc = '../data/DPP4_test_disguised.csv'

    tr_data = pd.read_csv(open(tr_loc, 'r'))
    val_data = pd.read_csv(open(val_loc, 'r'))

    features_names = np.intersect1d(tr_data.columns.values[2:],
                                    val_data.columns.values[2:])

    X_train, y_train = [np.array(tr_data[features_names], dtype='float32'),
                        np.array(tr_data['Act'], dtype='float32')]

    active_ix = np.array(X_train, dtype='bool').sum(0) >= 20
    X_train = X_train[:, active_ix]
    X_train = (X_train / X_train.max(0))

    print('Training data shape:(%d,%d)' % (X_train.shape))

    X_test, y_test = [np.array(val_data[features_names], dtype='float32'),
                      np.array(val_data['Act'], dtype='float32')]

    X_test = X_test[:, active_ix]
    X_test = (X_test / X_test.max(0))

    print('Test data shape:(%d,%d)' % (X_test.shape))

    ### Prepare the Graph Correlation matrix
    corr_mat = np.array(normalize(np.abs(np.corrcoef(X_train.transpose())),
                                  norm='l1', axis=1), dtype='float64')
    graph_mat = np.argsort(corr_mat, 1)[:, -num_neighbors:]

    ### 1 layer graph CNN
    #test = X_train[:5]
    #temp = graph_mat.astype(np.int32)
    #test = test.reshape(test.shape[0], test.shape[1], 1)
    #x_expanded = np.zeros((test.shape[0], test.shape[1], 5))
    #for i in range(len(test)):
    #    x_expanded[i, :, :] = test[i, :, :] * temp
    #x_expanded = tf.reshape(x_expanded, shape=(x_expanded.shape[0], x_expanded.shape[1], x_expanded.shape[2], 1))

    g_model = Sequential()
    g_model.add(GraphConv(filters=filters_1, neighbors_ix_mat=graph_mat,
                          num_neighbors=num_neighbors, activation='relu',
                          input_shape=(X_train.shape[1], 1)))
    g_model.add(Dropout(0.25))
    g_model.add(Flatten())
    g_model.add(Dense(1, kernel_regularizer=l2(0.01)))
    g_model.add(Dropout(0.1))

    g_model.summary()

    g_model.compile(loss='mean_squared_error', optimizer='adam')

    results['g'] = []
    for i in range(epochs):
        g_model.fit(X_train.reshape(X_train.shape[0], X_train.shape[1], 1), y_train,
                    epochs=1,
                    batch_size=batch_size,
                    verbose=0, )

        y_pred = g_model.predict(X_test.reshape(X_test.shape[0], X_test.shape[1], 1),
                                 batch_size=100).flatten()
        r_squared = (np.corrcoef(y_pred, y_test) ** 2)[0, 1]
        results['g'].append(r_squared)
        print('Epoch: %d, R squared: %.5f' % (i, r_squared))

    results['g'] = np.array(results['g'])
    print('1-Conv R squared = %.5f' % results['g'][-1])