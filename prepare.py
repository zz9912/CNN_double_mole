
import numpy as np
import tensorflow as tf
import scipy.io as scio
from sklearn.model_selection import train_test_split

seed = 313
np.random.seed(seed)
config = tf.compat.v1.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

def get_dataset(data_file, label_file, vec_num, test_size=0.2):
    dataFile1 = data_file
    dataFile2 = label_file
    #with codecs.open('train_data.mat', 'r', encoding='gb2312') as f:
    #    data = sio.loadmat(f)
    data = scio.loadmat(dataFile1)
    label = scio.loadmat(dataFile2)
    X = np.transpose(data['data'])
    Y = np.transpose(label['label'])
    # Generate the dataset
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size, random_state=0)
    X_train = X_train.reshape(np.size(X_train, 0), vec_num, 1)
    X_val = X_val.reshape(np.size(X_val, 0), vec_num, 1)
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    return X_train, X_val, Y_train, Y_val

def get_dataset_range(data_file, label_file, vec_num,test_size=0.2):
    dataFile1 = data_file
    dataFile2 = label_file
    data = scio.loadmat(dataFile1)
    label = scio.loadmat(dataFile2)
    X = np.transpose(data['data'])
    Y = np.transpose(label['label'])
    a = np.where((Y[:, 1] == 1) & (Y[:, 0] >= 2000) & (Y[:, 0] <= 16000))
    X = X[a]
    Y = Y[a]
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size, random_state=0)
    X_train = X_train.reshape(np.size(X_train, 0), vec_num, 1)
    X_val = X_val.reshape(np.size(X_val, 0), vec_num, 1)
    X_train = X_train.astype('float32')
    X_val = X_val.astype('float32')
    return X_train, X_val, Y_train, Y_val

