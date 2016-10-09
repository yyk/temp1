import data2 as data
from keras import backend as K
import sys
from keras.callbacks import ModelCheckpoint
import keras
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Input
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D, LSTM, GRU
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score)

np.random.seed(1337)  # for reproducibility

model_file = "./checkpoint"
# batch_size = 512
batch_size = 1024
# batch_size = 131072
nb_epoch = 10000
init='normal'

X_train, y_train, X_test, y_test = data.load_all()
nb_classes = 2

# Y_train = y_train
# Y_test = y_test
Y_train = np_utils.to_categorical(np.array(y_train), 2)
Y_test = np_utils.to_categorical(np.array(y_test), 2)

# input image dimensions
length = X_train.shape[2]
channel = X_train.shape[1]
print("samples: %d channel: %d length: %d" % (X_train.shape[0], channel, length))

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

print("x sample")
print(X_train[0])
print("y sample")
print(Y_train[0])

input_shape = X_train.shape[1:]
length = X_train.shape[1]
dimension = X_train.shape[2]
print("input_shape: %s" % (input_shape,))

# for i in range(10):
#     print(X_train[i])
#     print(Y_train[i])
# sys.exit(0)

model = Sequential()

model.add(Convolution1D(nb_filter=32, filter_length=1, activation='relu', border_mode='valid', init=init, input_dim=dimension, input_length=length))
# model.add(Convolution1D(nb_filter=32, filter_length=1, activation='relu', border_mode='valid'))
# model.add(Convolution1D(nb_filter=32, filter_length=1, activation='relu', border_mode='same', input_shape=input_shape))
# model.add(Convolution1D(nb_filter=64, filter_length=1, activation='relu', border_mode='same'))
# model.add(Convolution1D(nb_filter=64, filter_length=1, activation='relu', border_mode='same'))
# model.add(Convolution1D(nb_filter=128, filter_length=1, activation='relu', border_mode='same'))
# model.add(Convolution1D(nb_filter=128, filter_length=1, activation='relu', border_mode='same'))

# model.add(LSTM(4))
# model.add(TimeDistributed(LSTM(512)))
model.add(GRU(32, dropout_W=0.2, dropout_U=0.2, init=init, consume_less='gpu',
               # input_dim=dimension, input_length=length,
               return_sequences=True,
               ))
# model.add(GRU(32, init=init, return_sequences=True))
model.add(GRU(32, init=init))
# model.add(GRU(2048))

# model.add(Flatten())
model.add(Dense(256, init=init))
# model.add(Activation('relu'))
# model.add(Dropout(0.2))
# model.add(Dense(1))
# model.add(Dense(length))
# model.add(Activation('sigmoid'))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
# model.add(Dense(2))

model.compile(
        loss='binary_crossentropy',
         # loss='mse',
        # optimizer='adadelta',
        # optimizer='rmsprop',
        optimizer='adam',
        metrics=['accuracy'])

print(model.summary())

try:
    weights_to_load = "./checkpoint.backup"
    model.load_weights(weights_to_load)
    print("Loaded weights " + weights_to_load)
except Exception as e:
    print(e)

# checkpoint = ModelCheckpoint("./checkpoint", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
tensorboard = keras.callbacks.TensorBoard()
callbacks_list = [ tensorboard ]

highest_precision = 0
highest = (0,0,0,0)

# get_prediction = K.function(inputs=[model.layers[0].input, K.learning_phase()],
#                             outputs=[model.layers[-1].output])

def print_precision(class_number, predictions, y_test):
    percentages = [0.001, 0.01, 0.05, 0.10, 0.5]
    for percentage in percentages:
        percent = int(len(predictions) * percentage)
        p = predictions.argsort()[-percent:]
        selected_predictions = predictions[p]
        precision = np.count_nonzero(y_test[p]) / len(p)
        print("Class %d at top %1f%% precision %f, score %f" % (
        class_number, percentage * 100, precision, selected_predictions[0]))

for epoch in range(nb_epoch):
    # print(get_prediction(inputs=[X_test[:10], 0]))
    model.pop()
    y_pred = model.predict(X_test, batch_size=batch_size * 3).T
    print(y_pred[0][:20])
    print(y_pred[1][:20])

    print_precision(0, y_pred[0], y_test)
    print('-----------------------------')
    print_precision(1, y_pred[1], y_test)

    print('epoch {}'.format(str(epoch)))
    model.add(Activation('softmax'))
    model.fit(X_train, Y_train,
            batch_size=batch_size,
            nb_epoch=1,
            verbose=1,
            shuffle=True,
            validation_data=(X_test, Y_test),
            callbacks=callbacks_list
            # class_weight = {
            #     0: 2,
            #     1: 1.0}
          )
    model.save(model_file + ".backup", overwrite=True)
    continue

    # if (epoch+1) % 5 == 0:
    #     y_pred = model.predict_classes(X_train, batch_size=batch_size)
    #     accuracy = accuracy_score(y_train, y_pred)
    #     recall = recall_score(y_train, y_pred)
    #     precision = precision_score(y_train, y_pred)
    #     f1 = f1_score(y_train, y_pred)
    #     print('training set\n', 'Accuracy: {}\n'.format(accuracy), 'Recall: {}\n'.format(recall),
    #           'Precision: {}\n'.format(precision),
    #           'F1: {}'.format(f1))
    #
    # y_pred = model.predict_classes(X_test, batch_size=batch_size)
    # accuracy = accuracy_score(y_test, y_pred)
    # recall = recall_score(y_test, y_pred)
    # precision = precision_score(y_test, y_pred)
    # f1 = f1_score(y_test, y_pred)
    # print('test set\n', 'Accuracy: {}\n'.format(accuracy), 'Recall: {}\n'.format(recall),
    #       'Precision: {}\n'.format(precision),
    #       'F1: {}'.format(f1))
    #
    # # for category, (x, y) in tests.items():
    # #     score = model.evaluate(x, y, verbose=0, batch_size=batch_size)
    # #     print(' [%d] score: %f\taccuracy: %f' % (category, score[0], score[1]))
    # if precision > highest_precision and recall > 0.01:
    #     print("Precision increased from {} to {}, saving model to {}".format(highest_precision, precision, model_file))
    #     highest_precision = precision
    #     highest = (accuracy, recall, precision, f1)
    #     model.save(model_file + ".best", overwrite=True)
    # else:
    #     print("Precision didn't increase, current highest accuracy {} recall {} precision {} f1 {}".format(highest[0],
    #                                                                                                        highest[1],
    #                                                                                                        highest[2],
    #                                                                                                        highest[3]))
