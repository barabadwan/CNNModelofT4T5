import tensorflow as tf
import datetime
import os
from src.utils import *
from src.defineModels import *
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
epochs = 5
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

model = ln_model(input_shape=(size_t, size_x, n_c),
                     filter_shape=[filter_indicies_t, filter_indicies_x],
                     num_filter=num_filt,
                     sum_over_space=sum_over_space,
                     binarize = binarize)

# model = lnPool_model(input_shape=(size_t, size_x, n_c),
#                      filter_shape=[filter_indicies_t, filter_indicies_x],
#                      num_filter=num_filt,
#                      sum_over_space=sum_over_space,
#                      binarize = binarize,
#                      poolSize = (1,2))


######
# Fit Models
######
save_folder = data_set_folder+'/'+model.name + '/' + date_str + '/'
os.makedirs(save_folder)


saveCallbackFolder, saveCallback = makeSaveCallback(save_folder)

adamOpt = optimizers.Adam(lr= learningRate, decay=learningDecay)

model.compile(optimizer=adamOpt, loss='binary_crossentropy', metrics=['accuracy'])

hist = model.fit(train_in, np.squeeze(train_out), verbose=2, epochs=epochs, batch_size=batch_size,
                                     validation_data=(dev_in, np.squeeze(dev_out)), callbacks=[accCallback, saveCallback])

hist.saveFolder = save_folder

######
# Plot Results
######
getPlotLayerWeights(model, save_folder)
plotAccuracyLoss(hist, save_folder)
# PlotLearningRateSch(hist)

plt.show()