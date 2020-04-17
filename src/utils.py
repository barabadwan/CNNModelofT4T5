import numpy as np
import h5py as h
import matplotlib.pyplot as plt
import os
import tensorflow
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K


def load_data_rr(path, binarize = True):

    mat_contents = h.File(path, 'r')
    train_in = mat_contents['train_in'][:]
    train_out = mat_contents['train_out'][:]
    dev_in = mat_contents['dev_in'][:]
    dev_out = mat_contents['dev_out'][:]
    test_in = mat_contents['test_in'][:]
    test_out = mat_contents['test_out'][:]

    sample_freq = mat_contents['sampleFreq'][:]
    phase_step = mat_contents['phaseStep'][:]

    if binarize:
        train_out = train_out/abs(train_out)
        train_out = abs(train_out*(train_out>0))

        dev_out = dev_out/abs(dev_out)
        dev_out = abs(dev_out*(dev_out>0))

        test_out = test_out/abs(test_out)
        test_out = abs(test_out*(test_out>0))

    train_in = np.expand_dims(train_in, axis=3)
    dev_in = np.expand_dims(dev_in, axis=3)
    test_in = np.expand_dims(test_in, axis=3)

    train_out = np.expand_dims(train_out, axis=2)
    train_out = np.expand_dims(train_out, axis=3)
    dev_out = np.expand_dims(dev_out, axis=2)
    dev_out = np.expand_dims(dev_out, axis=3)
    test_out = np.expand_dims(test_out, axis=2)
    test_out = np.expand_dims(test_out, axis=3)

    mat_contents.close()

    return train_in, train_out, dev_in, dev_out, test_in, test_out, sample_freq, phase_step


def plotAccuracyLoss(hist, save_folder):
    modelName = hist.model.name
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    loss =  hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(len(acc))

    fig = plt.figure()
    plt.plot(epochs, acc, 'r')
    plt.plot(epochs, val_acc, 'b')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.title(f'Training and validation accuracy for {modelName}')
    fig.savefig(save_folder+'/Training and validation accuracy for '+modelName+'.pdf')

    fig = plt.figure()
    plt.plot(epochs, loss, 'r')
    plt.plot(epochs, val_loss, 'b')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.title(f'Training and validation loss for model {modelName}')
    fig.savefig(save_folder + '/Training and validation loss for ' + modelName+'.pdf')

def getPlotLayerWeights(model, save_folder):
    weights = []
    combFirstFilter = []
    combSecondFilter = []
    spatialDeg = 0
    for layer in model.layers[:-1]:
        if layer.get_weights():
            weights.append(layer.get_weights())
            layer_weights = layer.get_weights()
            T, X, n, fs = layer_weights[0].shape
            if model.name is not "conductance_model":
                fig = plt.figure()
                plt.title(layer.name)
                for f in range(fs):
                    plt.subplot(1, fs, f+1)
                    plt.suptitle(f'Filters for layer {layer.name} for {model.name}')
                    plt.imshow(np.flipud(layer_weights[0][:,:,0,f]), cmap = 'RdBu', extent = [0, 15, 300, 0])
                    fig.savefig(save_folder +'/Filters for '+ model.name+'.pdf')
            elif model.name == "conductance_model":
                firstFilter  = np.squeeze(np.squeeze(layer_weights[0][:, :, :, 0]))
                secondFilter = np.squeeze(np.squeeze(layer_weights[0][:, :, :, 1]))

                combFirstFilter.append(firstFilter)
                combSecondFilter.append(secondFilter)

                xx,tt = np.shape(combSecondFilter)
                if xx == 3:
                    fig = plt.figure()
                    plt.subplot(1, 2, 1)
                    plt.suptitle(f'Filters for combined layers of the {model.name}')
                    plt.imshow(np.transpose(combFirstFilter), cmap = 'RdBu', extent = [0, 15, 300, 0])
                    plt.subplot(1, 2, 2)
                    plt.imshow(np.transpose(combSecondFilter), cmap = 'RdBu', extent = [0, 15, 300, 0])
                    fig.savefig(save_folder +'/Filters for '+ model.name+'.pdf')
                    break


    return weights

def PlotLearningRateSch(history):
    plt.figure()
    plt.semilogx(history.history["lr"], history.history['loss'])
    plt.xlabel('Learning Rate'), plt.ylabel('Loss'), plt.title('Learning Rate Sweep')
    optimal_lr = history.history["lr"][history.history['loss'].index(min(history.history['loss']))]
    return optimal_lr

def makeSaveCallback(filename):
    checkpoint_path = filename
    checkpoint_filepath = checkpoint_path+"cp.ckpt"
    # Create a callback that saves the model's weights
    cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath,
                                                     save_weights_only=True,
                                                     verbose=1)
    return checkpoint_filepath, cp_callback


class accuracyThreshCallback(tensorflow.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.99):
            print("\nLoss is low so cancelling training!")
            self.model.stop_training = True

def l1_reg_sqrt(weight_matrix):
    return 0.1 * K.sum(K.sqrt(K.abs(weight_matrix)))


class BiasLayer(Layer):
    def __init__(self, **kwargs):
        super(BiasLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.bias = self.add_weight(name='bias',
                                    shape=(input_shape[-1],),
                                    initializer='zero',
                                    trainable=True)
        super(BiasLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return x + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape


