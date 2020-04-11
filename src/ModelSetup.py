import tensorflow as tf
import datetime
import os
from src.utils import *
from tensorflow.keras import optimizers

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
save_folder = data_set_folder + '/saved_parameters/' + date_str + '/'
os.makedirs(save_folder)


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
epochs = 50
batch_size = 128
learningRate = 0.01
learningDecay = 0

######
# Load data
######
train_in, train_out, dev_in, dev_out, test_in, test_out, sample_freq, phase_step = load_data_rr(path, binarize=binarize)

filter_indicies_t = int(np.ceil(filter_time*int(sample_freq)))
filter_indicies_x = int(np.ceil(filter_space/int(phase_step)))

m, size_t, size_x, n_c = train_in.shape

######
# Set callbacks, learning rate scheduler
######
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 1e-2 * 10**(epoch / 20))

accCallback = accuracyThreshCallback()


######
# Define models
######
def ln_model(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2,
                 sum_over_space=True, fit_reversal=True, weight_init=[], filter_weight_noise_std = 0.01, binarize = 1):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(num_filter, filter_shape, strides=(1, 1), name='First_Convolution',
                            kernel_initializer='glorot_normal', activation='relu', use_bias=False,
                            input_shape = input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(101, activation = 'sigmoid')
    ], name = 'LN_Model')

    return model


model = ln_model(input_shape=(size_t, size_x, n_c),
                     filter_shape=[filter_indicies_t, filter_indicies_x],
                     num_filter=num_filt,
                     sum_over_space=sum_over_space,
                     binarize = binarize)


######
# Fit Models
######
saveCallback = makeSaveCallback(save_folder+model.name)

adamOpt = optimizers.Adam(lr= learningRate, decay=learningDecay)

model.compile(optimizer=adamOpt, loss='binary_crossentropy', metrics=['accuracy'])

hist = model.fit(train_in, np.squeeze(train_out), verbose=2, epochs=epochs, batch_size=batch_size,
                                     validation_data=(dev_in, np.squeeze(dev_out)), callbacks=[accCallback])

######
# Plot Results
######
getPlotLayerWeights(model)
plotAccuracyLoss(hist)
# PlotLearningRateSch(hist)

plt.show()