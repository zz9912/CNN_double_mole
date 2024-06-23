import os
import tensorflow as tf
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from sklearn.model_selection import train_test_split
import numpy as np
from init.prepare import get_dataset,get_dataset_range
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from models.resnet import resnet_34_double_mole_double_batchatt
from tensorflow.keras.optimizers import Nadam

seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)
config = tf.compat.v1.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

# Initialization
data_weight = 5
batch_size = 32
epochs = 400
initial_lr = 1e-4

checkpoint_save_path= os.path.abspath(os.path.dirname(__file__)) + '/check_point/ours.ckpt'

class CustomSumLossCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        loss_sum = logs.get('val_output2_loss') + logs.get('val_output4_loss')
        logs['val_loss24'] = loss_sum

custom_sum_loss_callback = CustomSumLossCallback()

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=30,
                               verbose=2)

lr_reducer = ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=25,
                               verbose=2,
                               mode='min',
                               cooldown=0,
                               min_lr=1e-7)

model_checkpoint = ModelCheckpoint(checkpoint_save_path,
                                   monitor='val_loss24',
                                   save_weights_only=True,
                                   save_best_only=True,
                                   mode='min')

vec=287

data_path1 = os.path.abspath(os.path.dirname(__file__))+'/dataset/data_part1.mat'
label_path1 = os.path.abspath(os.path.dirname(__file__))+'/dataset/label_part1.mat'
data_path2 = os.path.abspath(os.path.dirname(__file__))+'/dataset/data_part2.mat'
label_path2 = os.path.abspath(os.path.dirname(__file__))+'/dataset/label_part2.mat'

X_train, X_val, Y_train, Y_val = get_dataset(data_path1, label_path1, vec_num=vec,test_size=0.2)
X_train1, X_val1, Y_train1, Y_val1 = get_dataset_range(data_path2, label_path2, vec_num=vec,test_size=0.2)

X_train = np.concatenate([X_train,X_train1],0)
X_val = np.concatenate([X_val,X_val1],0)
Y_train=np.concatenate([Y_train,Y_train1],0)
Y_val=np.concatenate([Y_val,Y_val1],0)

Y_train=np.log10(Y_train)

X_train, X_valk, Y_train, Y_valk = train_test_split(X_train, Y_train, test_size=0.1, random_state=0)

mode='rm'

# Train
model = resnet_34_double_mole_double_batchatt(vec,[2,2,2,2],mode)
model.compile(loss={'output1': 'mae',
                    'output2': 'mae',
                    'output3': 'mae',
                    'output4': 'mae'},
                  optimizer=Nadam(learning_rate=1e-3))
model.fit(X_train, {'output1':Y_train[:,0],'output2':Y_train[:,1],'output3':Y_train[:,0],'output4':Y_train[:,1]},
          batch_size=batch_size, epochs=epochs,
          verbose=1,
          validation_data=(X_valk, {'output1':Y_valk[:,0],'output2':Y_valk[:,1],'output3':Y_valk[:,0],'output4':Y_valk[:,1]}),
          callbacks=[custom_sum_loss_callback,early_stopping, lr_reducer, model_checkpoint])



# Evaluation
my_model = resnet_34_double_mole_double_batchatt(vec,[2,2,2,2],mode)
my_model.load_weights(checkpoint_save_path)
my_model.compile(loss={'output1': 'mae',
                    'output2': 'mae',
                       'output3': 'mae',
                       'output4': 'mae'},
                  optimizer=Nadam(learning_rate=1e-3))
u=0
k=len(X_val)
predicted = my_model.predict(X_val[u:k])
predicted_aux=predicted

predicted = np.array(predicted[0:4])
predicted=10**predicted
predicted_origin=predicted
Y_origin=Y_val[u:k,1].reshape(-1,1)
predictedg=predicted[2]
predictedh=predicted[3]

print(np.concatenate([Y_val[u:k,0].reshape(-1,1),predictedg,Y_val[u:k,1].reshape(-1,1),predictedh],axis=1))

scores = my_model.evaluate(X_val[u:k] ,[np.log10(Y_val[u:k,0]),np.log10(Y_val[u:k,1]),np.log10(Y_val[u:k,0]),np.log10(Y_val[u:k,1]),predicted_aux[4],predicted_aux[5],predicted_aux[6]], verbose=0)
print(my_model.metrics_names,scores)
MAEg = np.mean(abs(predictedg - Y_val[u:k,0].reshape(-1,1)))
MAEh = np.mean(abs(predictedh - Y_val[u:k,1].reshape(-1,1)))
SDg = np.std(abs(predictedg - Y_val[u:k,0].reshape(-1,1)))
SDh = np.std(abs(predictedh - Y_val[u:k,1].reshape(-1,1)))
R2h = 1-(np.sum((Y_origin - predictedh)**2) / np.sum((Y_origin - np.mean(Y_origin))**2))
print('\nprediction MAEg: ', MAEg)
print('\nprediction SDg: ', SDg)
print('\nprediction MAEh: ', MAEh)
print('\nprediction SDh: ', SDh)
print('\nprediction R2h: ', R2h)


























































