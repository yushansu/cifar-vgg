from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import optimizers
import numpy as np
from keras.layers.core import Lambda
from keras import backend as K
from keras import regularizers
from keras.utils import np_utils
import tensorflow as tf

class cifar10vgg:
    def __init__(self,train=False):
        self.num_classes = 10
        self.weight_decay = 0.0005
        self.x_shape = [32,32,3]

        self.model = self.build_model()
        if train:
            self.model = self.train(self.model)
        else:
            self.model.load_weights('cifar10vgg.h5')


    def build_model(self):
        # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = self.weight_decay

        model.add(Conv2D(64, (3, 3), padding='same',
                         input_shape=self.x_shape,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))


        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes))
        model.add(Activation('softmax'))
        return model


    def normalize(self,X_train,X_test):
        #this function normalize inputs for zero mean and unit variance
        # it is used when training a model.
        # Input: training set and test set
        # Output: normalized training set and test set according to the trianing set statistics.
        mean = np.mean(X_train,axis=(0,1,2,3))
        std = np.std(X_train, axis=(0, 1, 2, 3))
        X_train = (X_train-mean)/(std+1e-7)
        X_test = (X_test-mean)/(std+1e-7)
        return X_train, X_test

    def normalize_production(self,x):
        #this function is used to normalize instances in production according to saved training set statistics
        # Input: X - a training set
        # Output X - a normalized training set according to normalization constants.

        #these values produced during first training and are general for the standard cifar10 training set normalization
        mean = 120.707
        std = 64.15
        return (x-mean)/(std+1e-7)

    def predict(self,x,normalize=True,batch_size=50):
        if normalize:
            x = self.normalize_production(x)
        return self.model.predict(x,batch_size)

    def train(self,model):

        #training parameters
        batch_size = 128
        maxepoches = 250
        learning_rate = 0.1
        lr_decay = 1e-6
        lr_drop = 20
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train, x_test = self.normalize(x_train, x_test)

        y_train = keras.utils.np_utils.to_categorical(y_train, self.num_classes)
        y_test = keras.utils.np_utils.to_categorical(y_test, self.num_classes)

        def lr_scheduler(epoch):
            return learning_rate * (0.5 ** (epoch // lr_drop))
        reduce_lr = keras.callbacks.LearningRateScheduler(lr_scheduler)

        #data augmentation
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        #optimization details
        sgd = tf.keras.optimizers.SGD(lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])

        # training process in a for loop with learning rate drop every 25 epoches.

        historytemp = model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=maxepoches,
                            validation_data=(x_test, y_test),callbacks=[reduce_lr],verbose=2)
        model.save_weights('cifar10vgg.h5')
        return model

if __name__ == '__main__':

    # prepare the following files
    SOFTMAX_INPUTS_TRAIN_PATH = 'cifar10_softmax_inputs_train.npy'
    SOFTMAX_OUTPUTS_TRAIN_PATH = 'cifar10_softmax_outputs_train.npy'
    SOFTMAX_INPUTS_TEST_PATH = 'cifar10_softmax_inputs_test.npy'
    SOFTMAX_OUTPUTS_TEST_PATH = 'cifar10_softmax_outputs_test.npy'
    SOFTMAX_W_PATH = 'cifar10_softmax_W.npy'
    SOFTMAX_B_PATH = 'cifar10_softmax_b.npy'
    LABELS_TRAIN_PATH = 'cifar10_labels_train.npy'
    LABELS_TEST_PATH = 'cifar10_labels_test.npy'

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    y_train = keras.utils.np_utils.to_categorical(y_train, 10)
    y_test = keras.utils.np_utils.to_categorical(y_test, 10)

    # inference pass on test set
    model = cifar10vgg()

    predicted_x = model.predict(x_test)
    residuals = np.argmax(predicted_x,1)!=np.argmax(y_test,1)

    loss = sum(residuals)/len(residuals)
    print("the validation 0/1 loss is: ",loss)

        
    vgg_model = model.model
    print(vgg_model.summary())
    # model.build_model().save('/Users/suyushan/Documents/cifar-vgg/saved_model')

    # prepare inputs (X)
    cifar10_softmax_in_layer = vgg_model.layers[57]
    keras_function_inputs = K.function([vgg_model.input], [cifar10_softmax_in_layer.output])
    # cifar10_softmax_inputs_test (X_test) (10000 * 512)
    inputs_test = keras_function_inputs([x_test])
    with open(SOFTMAX_INPUTS_TEST_PATH, 'wb') as f:
        np.save(f, inputs_test[0])
    # cifar10_softmax_inputs_train (X_train) (50000 * 512)
    for i in range(0, len(x_train), 250):
        inputs_train_i = keras_function_inputs([x_train[i:i+250]])
        if i == 0:
            inputs_train = inputs_train_i[0]
        else:
            inputs_train = np.concatenate((inputs_train, inputs_train_i[0]))
    with open(SOFTMAX_INPUTS_TRAIN_PATH, 'wb') as f:
        np.save(f, inputs_train)

    # prepare outputs (Y) and labels (lbls_train, lbls_test)
    cifar10_softmax_out_layer = vgg_model.layers[58]
    keras_function_outputs = K.function([vgg_model.input], [cifar10_softmax_out_layer.output])
    # cifar10_softmax_outputs_test (Y_test) (10000 * 10)
    outputs_test = keras_function_outputs([x_test])
    with open(SOFTMAX_OUTPUTS_TEST_PATH, 'wb') as f:
        np.save(f, outputs_test[0])
    # cifar10_labels_test (lbls_test) (10000 * 1)
    with open(LABELS_TEST_PATH, 'wb') as f:
        np.save(f, np.argmax(outputs_test[0], axis=1).astype(np.int32))
    # cifar10_softmax_outputs_train (Y_train) (50000 * 10)
    for i in range(0, len(x_train), 250):
        outputs_train_i = keras_function_outputs([x_train[i:i+250]]) 
        if i == 0:
            outputs_train = outputs_train_i[0]
        else:
            outputs_train = np.concatenate((outputs_train, outputs_train_i[0]))
    with open(SOFTMAX_OUTPUTS_TRAIN_PATH, 'wb') as f:
        np.save(f, outputs_train)
    # cifar10_labels_train (lbls_train) (50000 * 1)
    with open(LABELS_TRAIN_PATH, 'wb') as f:
        np.save(f, np.argmax(outputs_train, axis=1).astype(np.int32))

    # prepare weight and bias (W and b)
    dense_layer_weights = cifar10_softmax_out_layer.get_weights()
    weight = dense_layer_weights[0]
    bias = dense_layer_weights[1]
    with open(SOFTMAX_W_PATH, 'wb') as f:
        np.save(f, weight)
    with open(SOFTMAX_B_PATH, 'wb') as f:
        np.save(f, bias)

