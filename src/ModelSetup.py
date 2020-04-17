import tensorflow as tf
import datetime
import os
from src.utils import *
from src.defineModels import *
from tensorflow.keras import optimizers
import easygui
######
# Set data paths
######
# define the input path
# data set location
data_set_folder = os.getcwd()
# save in a folder with the date
date_str = str(datetime.datetime.now())
date_str = '_'.join(date_str.split(' '))
date_str = '-'.join(date_str.split(':'))


data_set_names = ['xtPlot_ns20_xe360_xs360_ye100_ys5_pe360_ps5_sf100_tt1_nt16_hl0-2_vs100_df0-05_no0_csInf_sy0.mat',
                  'xtPlot_sineWaves_sl20_ll90_pe360_ps5_sf100_tt1_nt6080_hl0-2_vs100_df0-05_no0_csInf.mat',
                  'xtPlot_ns20_xe360_xs360_ye100_ys5_pe360_ps5_sf100_tt1_nt16_hl0-2_vs100_df0-05_no0_csInf_sy1.mat']


image_type_idxes = [0]
image_types = ['nat', 'sine', 'synth']

for image_type_idx in image_type_idxes:
    path = data_set_folder + '/Data/natural_images/xt/' + data_set_names[image_type_idx]

######
# Set run parameters
######
num_runs = 1
filter_time = 0.3  # s
noise_std_list = [0.1]
filter_space = 15  # degrees
sum_over_space = [False]
num_filt = 2
binarize = True
epochs = 100
batch_size = 128
learningRate = 0.1
learningDecay = 10

######
# Load data
######
train_in, train_out, dev_in, dev_out, test_in, test_out, sample_freq, phase_step = load_data_rr(path, binarize=binarize)
######
# Data_in had dims (Samples, Time, Space)
######
# Set up filter lengths based on the entered filter time and space and the associated phase/freq
filter_indicies_t = int(np.ceil(filter_time*int(sample_freq)))
filter_indicies_x = int(np.ceil(filter_space/int(phase_step)))


# Validation data processing
# Validation data intially in (Samples, Velocities at each time, Space =1, 1)
# We want to crop the first chunk in time as the filter needs a certain amount of points in the past to work
# The time vector should be cropped because we can't make predictions on points where we have no access to the past
# The space vector should be retiled

m, size_t, size_x, n_c = train_in.shape

pad_t = int((filter_indicies_t- 1))
pad_x = int((filter_indicies_x- 1))

train_out = np.tile(train_out, (1, 1, size_x - pad_x, 1))
dev_out = np.tile(dev_out, (1, 1, size_x - pad_x, 1))
test_out = np.tile(test_out, (1, 1, size_x - pad_x, 1))

train_out = train_out[:, pad_t:, :, :]
dev_out = dev_out[:, pad_t:, :, :]
test_out = test_out[:, pad_t:, :, :]

######
# Set callbacks, learning rate scheduler
######
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-2 * 10**(epoch / 20))

accCallback = accuracyThreshCallback()


######
# Define models
######

LNModel = ln_model(input_shape=(size_t, size_x, n_c),
                     filter_shape=[filter_indicies_t, filter_indicies_x],
                     num_filter=num_filt,
                     sum_over_space=sum_over_space,
                     binarize = binarize)


LNPoolModel = lnPool_model(input_shape=(size_t, size_x, n_c),
                     filter_shape=[filter_indicies_t, filter_indicies_x],
                     num_filter=num_filt,
                     sum_over_space=sum_over_space,
                     binarize = binarize,
                     poolSize = (1,2))

LNLNModel = lnln_model(input_shape=(size_t, size_x, n_c),
                     filter_shape=[filter_indicies_t, filter_indicies_x],
                     num_filter=num_filt,
                     sum_over_space=sum_over_space,
                     binarize = binarize)

conductanceModel = conductance_model(input_shape=(size_t, size_x, n_c),
                     filter_shape=[filter_indicies_t, filter_indicies_x],
                     num_filter=num_filt,
                     sum_over_space=sum_over_space,
                     binarize = binarize) # worth pointing out, the spatial filter size is reset in the model

anatomicalModel = anatomical_model(input_shape=(size_t, size_x, n_c),
                     filter_shape=[filter_indicies_t, filter_indicies_x],
                     num_filter=num_filt,
                     sum_over_space=sum_over_space,
                     binarize = binarize) # worth pointing out, the spatial filter size is reset in the model
######
# Fit Models
######
def fitModel(model,retrain=0):
    if retrain:
        save_folder = data_set_folder+'/'+model.name + '/' + date_str + '/'
        os.makedirs(save_folder)


        saveCallbackFolder, saveCallback = makeSaveCallback(save_folder)

        adamOpt = optimizers.Adam(lr= learningRate, decay=learningDecay)

        SGD =  optimizers.SGD(learning_rate=learningRate, nesterov=True)
        rmsProp = optimizers.RMSprop(learning_rate=0.01)

        model.compile(optimizer= SGD, loss='binary_crossentropy', metrics=['accuracy'])

        hist = model.fit(train_in, np.squeeze(train_out), verbose=2, epochs=epochs, batch_size=batch_size,
                         validation_data=(dev_in, np.squeeze(dev_out)), callbacks=[accCallback, saveCallback])
        model.save(save_folder+'savedmodel.h5')
    else:

        path = easygui.fileopenbox(default = os.getcwd())
        save_folder = os.path.dirname(path)
        hist = tf.keras.models.load_model(path)


    getPlotLayerWeights(model, save_folder)
    plotAccuracyLoss(hist, save_folder)

    return


######
# Plot Results
######

# fitModel(LNModel,1)
# fitModel(LNPoolModel, 1)
# fitModel(conductanceModel,1)
fitModel(LNLNModel,1)
fitModel(LNLNModel,1)
fitModel(LNLNModel,1)

# fitModel(anatomicalModel)
# PlotLearningRateSch(hist)


plt.show()