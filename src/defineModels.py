import tensorflow as tf

def ln_model(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2,
                 sum_over_space=True, fit_reversal=True, weight_init=[], filter_weight_noise_std = 0.01, binarize = 1):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(num_filter, filter_shape, strides=(1, 1), name='First_Convolution',
                            kernel_initializer='glorot_normal', use_bias=False,
                            input_shape = input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(101, activation = 'sigmoid')
    ], name = 'LN_Model')

    return model



def lnPool_model(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2,
                 sum_over_space=True, fit_reversal=True, weight_init=[],
                 filter_weight_noise_std = 0.01, binarize = 1, poolSize = (1,2)):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(num_filter, filter_shape, strides=(1, 1), name='First_Convolution',
                            kernel_initializer='glorot_normal', activation='relu', use_bias=False,
                            input_shape = input_shape),
        tf.keras.layers.AveragePooling2D(poolSize),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(101, activation = 'sigmoid')
    ], name = 'lnPool_model')

    return model

