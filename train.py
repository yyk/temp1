import data2 as data
from keras.callbacks import ModelCheckpoint
import keras
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, Convolution1D, LSTM, GRU
from keras.utils import np_utils
import numpy as np
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, accuracy_score)

np.random.seed(1337)  # for reproducibility

model_file = "./checkpoint"
batch_size = 2048
# batch_size = 8192
nb_epoch = 10000

X_train, Y_train, X_test, Y_test = data.load_all()
nb_classes = 2

# input image dimensions
length = X_train.shape[2]
channel = X_train.shape[1]
print("samples: %d channel: %d length: %d" % (X_train.shape[0], channel, length))

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

input_shape = X_train.shape[1:]
print("input_shape: %s" % (input_shape,))

model = Sequential()

model.add(Convolution1D(nb_filter=64, filter_length=1, activation='relu', border_mode='same', input_shape=input_shape))
model.add(Convolution1D(nb_filter=128, filter_length=1, activation='relu', border_mode='same', input_shape=input_shape))
model.add(Convolution1D(nb_filter=128, filter_length=1, activation='relu', border_mode='same', input_shape=input_shape))

model.add(LSTM(2048))
# model.add(TimeDistributed(LSTM(512)))
# model.add(LSTM(512))
# model.add(LSTM(2048))
# model.add(GRU(2048))

# model.add(Dense(8192))
# model.add(Activation('relu'))
# model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(
        loss='binary_crossentropy',
        optimizer='adadelta',
        metrics=['accuracy'])

# model.load_weights("./checkpoint")

# checkpoint = ModelCheckpoint("./checkpoint", monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# callbacks_list = [checkpoint]

highest_precision = 0
highest = (0,0,0,0)
for epoch in range(nb_epoch):
    model.fit(X_train, Y_train,
            batch_size=batch_size,
            nb_epoch=1,
            verbose=1,
            validation_data=(X_test, Y_test),
            # callbacks=callbacks_list)
            # class_weight = {
            #     0: 2,
            #     1: 1.0}
          )
    print('epoch {}'.format(str(epoch)))

    if (epoch+1) % 5 == 0:
        y_pred = model.predict_classes(X_train, batch_size=batch_size)
        accuracy = accuracy_score(Y_train, y_pred)
        recall = recall_score(Y_train, y_pred)
        precision = precision_score(Y_train, y_pred)
        f1 = f1_score(Y_train, y_pred)
        print('training set\n', 'Accuracy: {}\n'.format(accuracy), 'Recall: {}\n'.format(recall),
              'Precision: {}\n'.format(precision),
              'F1: {}'.format(f1))

    y_pred = model.predict_classes(X_test, batch_size=batch_size)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print('test set\n', 'Accuracy: {}\n'.format(accuracy), 'Recall: {}\n'.format(recall),
          'Precision: {}\n'.format(precision),
          'F1: {}'.format(f1))

    # for category, (x, y) in tests.items():
    #     score = model.evaluate(x, y, verbose=0, batch_size=batch_size)
    #     print(' [%d] score: %f\taccuracy: %f' % (category, score[0], score[1]))
    if precision > highest_precision and recall > 0.01:
        print("Precision increased from {} to {}, saving model to {}".format(highest_precision, precision, model_file))
        highest_precision = precision
        highest = (accuracy, recall, precision, f1)
        model.save(model_file, overwrite=True)
    else:
        print("Precision didn't increase, current highest accuracy {} recall {} precision {} f1 {}".format(highest[0],
                                                                                                           highest[1],
                                                                                                           highest[2],
                                                                                                           highest[3]))
