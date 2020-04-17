import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.python.keras.utils.data_utils import *
import numpy as np
from tensorflow.keras import backend as K
from src.utils import *


def l_model(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2,
                 sum_over_space=True, fit_reversal=True, weight_init=[], filter_weight_noise_std = 0.01, binarize = 1):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(num_filter, filter_shape, strides=(1, 1), name='First_Convolution',
                            kernel_initializer='he_uniform', use_bias=False,
                            input_shape = input_shape),
        tf.keras.layers.Conv2D(1, (1, 1), strides=(1, 1), name='conv2',
               kernel_initializer='he_uniform',
               use_bias=False, dtype='float32')
    ], name = 'L_Model')

    return model

def ln_model(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2,
                 sum_over_space=True, fit_reversal=True, weight_init=[], filter_weight_noise_std = 0.01, binarize = 1):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(num_filter, filter_shape, strides=(1, 1), name='First_Convolution',
                            kernel_initializer='he_uniform', use_bias=False, activation='relu',
                            input_shape = input_shape),
        # tf.keras.layers.Flatten(),
        tf.keras.layers.Conv2D(1, (1, 1), strides=(1, 1), name='CombineT4T5',
                               kernel_initializer='he_uniform',
                               use_bias=False, dtype='float32', activation = 'sigmoid')
    ], name = 'LN_Model')

    return model

def lnPool_model(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2,
                 sum_over_space=True, fit_reversal=True, weight_init=[],
                 filter_weight_noise_std = 0.01, binarize = 1, poolSize = (1,2)):

    poolSize = (1, input_shape[2])
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(num_filter, filter_shape, strides=(1, 1), name='First_Convolution',
                            kernel_initializer='he_uniform', activation='relu', use_bias=False,
                            input_shape = input_shape),
        tf.keras.layers.AveragePooling2D(poolSize),
        tf.keras.layers.Conv2D(1, (1, 1), strides=(1, 1), name='CombineT4T5',
                               kernel_initializer='he_uniform',
                               use_bias=False, dtype='float32', activation='sigmoid')
    ], name = 'LNPool_model')

    return model


def lnln_model(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2,
                 sum_over_space=True, fit_reversal=True, weight_init=[],
                 filter_weight_noise_std = 0.01, binarize = 1):
    inputs = Input(input_shape)


    DSFilter = Conv2D(num_filter, filter_shape, strides=(1, 1), name='Space_time_Convolution',
                        kernel_initializer='glorot_uniform', activation='relu', use_bias=False)(inputs)

    filter_shape2 = (filter_shape[0], 1)
    nonDSFilter = Conv2D(1, filter_shape2, strides=(1, 1), name='Time_Convolution',
                        kernel_initializer='glorot_uniform', activation='relu', use_bias=False)(inputs)




    dLN = Lambda(lambda lam: lam[0]*lam[1])([DSFilter,nonDSFilter[:,:,1:-1,:]])

    CombT4T5 = Conv2D(1, (1, 1), strides=(1, 1), name='CombineT4T5',
                      kernel_initializer='he_uniform',
                      use_bias=False, dtype='float32', activation='sigmoid')(dLN)

    model = Model(inputs=inputs, outputs=CombT4T5, name='LNLN_model')



    return model



def conductance_model(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2,
                 sum_over_space=True, fit_reversal=False, weight_init=[],
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
                kernel_initializer='he_normal',
                activation='relu', kernel_regularizer= l1_reg_sqrt)

    c2 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='g2',
                kernel_initializer='he_normal',
                activation='relu', kernel_regularizer= l1_reg_sqrt)

    c3 = Conv2D(num_filter, filter_shape, strides=(1, 1), name='g3',
                kernel_initializer='he_normal',
                activation='relu', kernel_regularizer= l1_reg_sqrt)

    g1 = c1(s1)
    g2 = c2(s2)
    g3 = c3(-s3)
    # if fit_reversal:
    #     expand_last = Lambda(lambda lam: K.expand_dims(lam, axis=-1))
    #     squeeze_last = Lambda(lambda lam: K.squeeze(lam, axis=-1))
    #
    #     g1 = expand_last(g1)
    #     g2 = expand_last(g2)
    #     g3 = expand_last(g3)
    #
    #     numerator_in = Lambda(lambda inputs: K.concatenate(inputs, axis=4))([g1, g2, g3])
    #     numerator = Conv3D(1, (1, 1, 1), strides=(1, 1, 1), name='create_numerator',
    #                        kernel_initializer='glorot_uniform',
    #                        kernel_regularizer=l1_reg_sqrt,
    #                        use_bias=False)(numerator_in)
    #
    #     denominator = Lambda(lambda inputs: g_leak + inputs[0] + inputs[1] + inputs[2])([g1, g2, g3])
    #     vm = Lambda(lambda inputs: inputs[0] / inputs[1])([numerator, denominator])
    #     vm = squeeze_last(vm)
    # else:
    g1_v_inh = Lambda(lambda lam: lam * v_inh)(g1)
    g2_v_exc = Lambda(lambda lam: lam * v_exc)(g2)
    g3_v_inh = Lambda(lambda lam: lam * v_inh)(g3)

    numerator = Lambda(lambda inputs: inputs[0] + inputs[1] + inputs[2])([g1_v_inh, g2_v_exc, g3_v_inh])
    denominator = Lambda(lambda inputs: g_leak + inputs[0] + inputs[1] + inputs[2])([g1, g2, g3])
    vm = Lambda(lambda inputs: inputs[0] / inputs[1])([numerator, denominator])

    # vm_bias = BiasLayer()(vm)
    # vm_rect = Lambda(lambda lam: K.relu(lam))(vm_bias)

    # FlattenVM = tf.keras.layers.Flatten()(vm)
    # dropFlat = Dropout(0.2)()

    combinedT4T5 = Conv2D(1, (1, 1), strides=(1, 1), name='conv2',
                           kernel_initializer='he_uniform',
                           use_bias=False, dtype='float32', activation='sigmoid')(vm)



    model = Model(inputs=inputs, outputs=combinedT4T5, name='conductance_model')

    return model


def anatomical_model(input_shape=(11, 9, 1), filter_shape=(21, 9), num_filter=2,
                 sum_over_space=True, fit_reversal=True, weight_init=[],
                 filter_weight_noise_std = 0.01, binarize = 1):

    inputs = Input(input_shape)
    filter_shape[1] = 3

    LamFilters = 1
    MedFilters = 1

    T4T5Units = 3

    s1 = Lambda(lambda lam: lam[:, :, 0:-2, :])(inputs)
    s2 = Lambda(lambda lam: lam[:, :, 1:-1, :])(inputs)
    s3 = Lambda(lambda lam: lam[:, :, 2:, :])(inputs)

    LL1 = Conv2D(LamFilters, filter_shape, strides=(1, 1), name='LL1',
                kernel_initializer='he_uniform',
                activation='relu')(s1)

    ML1 = Dense(8)(LL1)

    LL2 = Conv2D(LamFilters, filter_shape, strides=(1, 1), name='LL2',
                kernel_initializer='he_uniform',
                activation='relu')(s2)

    ML2 = Dense(8)(LL2)

    LL3 = Conv2D(LamFilters, filter_shape, strides=(1, 1), name='LL3',
                kernel_initializer='he_uniform',
                activation='relu')(s3)

    ML3 = Dense(8)(LL3)


    layersIn = [ML1, ML2,  ML3]
    # layersIn = [LL1, LL2, LL3]

    T4T5Ins = concatenate(layersIn, name = 'concatenate')

    # T4T5Ins = Lambda(lambda inputs: (-inputs[0]  +inputs[1] - inputs[2])/(1+inputs[0]  +inputs[1] + inputs[2]))(layersIn)
    T4T5Layer = Dense(4, activation='relu')(T4T5Ins)


    FlattenVM = tf.keras.layers.Flatten()(T4T5Layer)
    # dropFlat = Dropout(0.2)()
    DenseF =  tf.keras.layers.Dense(101, activation = 'sigmoid')(FlattenVM)

    model = Model(inputs=inputs, outputs=DenseF, name='anatomical_model')

    return model