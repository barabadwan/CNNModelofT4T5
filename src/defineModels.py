import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

def l_model(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2,
                 sum_over_space=True, fit_reversal=True, weight_init=[], filter_weight_noise_std = 0.01, binarize = 1):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(num_filter, filter_shape, strides=(1, 1), name='First_Convolution',
                            kernel_initializer='glorot_normal', use_bias=False,
                            input_shape = input_shape),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(101, activation = 'sigmoid')
    ], name = 'L_Model')

    return model

def ln_model(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2,
                 sum_over_space=True, fit_reversal=True, weight_init=[], filter_weight_noise_std = 0.01, binarize = 1):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(num_filter, filter_shape, strides=(1, 1), name='First_Convolution',
                            kernel_initializer='glorot_normal', use_bias=False, activation='relu',
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
    ], name = 'LNPool_model')

    return model


def lnln_model(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2,
                 sum_over_space=True, fit_reversal=True, weight_init=[],
                 filter_weight_noise_std = 0.01, binarize = 1):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(num_filter, filter_shape, strides=(1, 1), name='First_Convolution',
                            kernel_initializer='glorot_uniform', activation='relu', use_bias=False,
                            input_shape = input_shape),
        tf.keras.layers.Conv2D(num_filter, filter_shape, strides=(1, 1), name='Second_Convolution',
                               kernel_initializer='glorot_uniform', activation='relu', use_bias=False),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(101, activation = 'sigmoid')
    ], name = 'LNLN_model')

    return model



def conductance_model(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2,
                 sum_over_space=True, fit_reversal=True, weight_init=[],
                 filter_weight_noise_std = 0.01, binarize = 1):

    inputs = Input(input_shape)
    filter_shape[1] = 1

    v_leak = 0
    v_exc = 60
    v_inh = -30
    g_leak = 1

    s1 = Lambda(lambda lam: lam[:, :, 0:-2, :])(inputs)
    s2 = Lambda(lambda lam: lam[:, :, 1:-1, :])(inputs)
    s3 = Lambda(lambda lam: lam[:, :, 2:, :])(inputs)

    c1 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='g1',
                kernel_initializer='glorot_normal',
                activation='relu')

    c2 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='g2',
                kernel_initializer='glorot_normal',
                activation='relu')

    c3 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='g3',
                kernel_initializer='glorot_normal',
                activation='relu')

    g1 = c1(s1)
    g2 = c2(s2)
    g3 = c3(s3)

    g1_v_inh = Lambda(lambda lam: lam * v_inh)(g1)
    g2_v_exc = Lambda(lambda lam: lam * v_exc)(g2)
    g3_v_inh = Lambda(lambda lam: lam * v_inh)(g3)

    numerator = Lambda(lambda inputs: inputs[0] + inputs[1] + inputs[2])([g1_v_inh, g2_v_exc, g3_v_inh])
    denominator = Lambda(lambda inputs: g_leak + inputs[0] + inputs[1] + inputs[2])([g1, g2, g3])
    vm = Lambda(lambda inputs: inputs[0] / inputs[1])([numerator, denominator])


    FlattenVM = tf.keras.layers.Flatten()(vm)
    # dropFlat = Dropout(0.2)()
    DenseF =  tf.keras.layers.Dense(101, activation = 'sigmoid')(FlattenVM)

    model = Model(inputs=inputs, outputs=DenseF, name='conductance_model')

    return model