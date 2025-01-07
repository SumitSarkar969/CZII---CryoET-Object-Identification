import keras._tf_keras.keras as keras
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, GlobalAveragePooling3D
from keras._tf_keras.keras.layers import Dropout, concatenate, multiply, Dense, GroupNormalization
from keras._tf_keras.keras.layers import LeakyReLU
from keras._tf_keras.keras.optimizers import Adam
from keras.src.metrics import iou_metrics, F1Score
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.initializers import GlorotNormal
from tensorflow.python.keras.metrics import MeanIoU, accuracy


class InstanceNormalization(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.norm = GroupNormalization(groups=-1)
    def call(self, x):
        return self.norm(x)

# Squeeze-and-Excitation layer
class SE_Layer(keras.layers.Layer):
    def __init__(self, ch, ratio = 16, **kwargs):
        super(SE_Layer, self).__init__(**kwargs)
        self.gl = GlobalAveragePooling3D(keepdims=True)
        self.fc1 = Dense(ch//ratio, activation='relu')
        self.fc2 = Dense(ch, activation='sigmoid')
    def call(self, input_block):
        x = self.gl(input_block)
        x = self.fc1(x)
        x = self.fc2(x)
        return multiply([input_block, x])

# Model
def My_LATUP(input_shape: tuple, loss)->keras.Model:
    inputs = Input(input_shape)

    # Encoder Block 1 (Parallel Convolutions Block) (E1)
    e1_pc_embed = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), activation=LeakyReLU(negative_slope=0.1), padding='same', name='E1_PC_Embedded_Layer')(inputs)
    e1_pc_conv1 = Conv3D(32, (1, 1, 1), strides=(1, 1, 1), activation=LeakyReLU(negative_slope=0.1), padding='same', name='E1_PC_Conv1_Layer')(e1_pc_embed)
    e1_pc_conv2 = Conv3D(32, (3, 3, 3), strides=(1, 1, 1), activation=LeakyReLU(negative_slope=0.1), padding='same', name='E1_PC_Conv2_Layer')(e1_pc_embed)
    e1_pc_conv3 = Conv3D(32, (5, 5, 5), strides=(1, 1, 1), activation=LeakyReLU(negative_slope=0.1), padding='same', name='E1_PC_Conv3_Layer')(e1_pc_embed)

    e1_pc_maxpool1 = MaxPooling3D(pool_size=(2, 2, 2), name='E1_maxpool1_Layer')(e1_pc_conv1)
    e1_pc_maxpool2 = MaxPooling3D(pool_size=(2, 2, 2), name='E1_maxpool2_Layer')(e1_pc_conv2)
    e1_pc_maxpool3 = MaxPooling3D(pool_size=(2, 2, 2), name='E1_maxpool3_Layer')(e1_pc_conv3)

    e1_pc_concat = concatenate([e1_pc_maxpool1, e1_pc_maxpool2, e1_pc_maxpool3], name='E1_concat_Layer')

    #Encoder Block 2 (E2)
    e2_se1 = SE_Layer(96, ratio=8, name='E2_SE1_Layer')(e1_pc_concat)
    e2_conv1 = Conv3D(64, (3, 3, 3), activation=LeakyReLU(negative_slope=0.1), padding='same', kernel_regularizer=l2(0.02), name='E2_Conv1_Layer')(e2_se1)
    e2_instance = InstanceNormalization(name='E2_instance_Layer')(e2_conv1)
    e2_conv2 = Conv3D(64, (3, 3, 3), activation=LeakyReLU(negative_slope=0.1), padding='same', kernel_regularizer=l2(0.02),name='E2_Conv2_Layer')(e2_instance)
    e2_dropout = Dropout(0.2, name='E2_Drop')(e2_conv2)
    e2_maxpool1 = MaxPooling3D(pool_size=(2, 2, 2), name='E2_maxpool1_Layer')(e2_dropout)

    #Encoder Block 3 (E3)
    e3_se1 = SE_Layer(64, ratio=8, name='E3_SE1_Layer')(e2_maxpool1)
    e3_conv1 = Conv3D(128, (3, 3, 3), activation=LeakyReLU(negative_slope=0.1), padding='same', kernel_regularizer=l2(0.02), name='E3_Conv1_Layer')(e3_se1)
    e3_instance = InstanceNormalization(name='E3_instance_Layer')(e3_conv1)
    e3_conv2 = Conv3D(128, (3, 3, 3), activation=LeakyReLU(negative_slope=0.1), padding='same', kernel_regularizer=l2(0.02), name='E3_Conv2_Layer')(e3_instance)
    e3_dropout = Dropout(0.2, name='E3_drop')(e3_conv2)
    e3_maxpool1 = MaxPooling3D(pool_size=(2, 2, 2), name='E3_maxpool1_Layer')(e3_dropout)

    bn_se1 = SE_Layer(128, ratio=8, name='BN_SE1_Layer')(e3_maxpool1)

    #Decoder Block 3 (D3)
    d3_up = UpSampling3D(size=(2, 2, 2), name='D3_up')(bn_se1)
    d3_conv1 = Conv3D(128, (3, 3, 3), activation=LeakyReLU(negative_slope=0.1), padding='same', kernel_regularizer=l2(0.02), name='D3_Conv1_Layer')(d3_up)
    d3_instance = InstanceNormalization(name='D3_instance_Layer')(d3_conv1)
    d3_concat = concatenate([d3_instance, e3_dropout], name='D3_concat_Layer')
    d3_conv2 = Conv3D(128, (3, 3, 3), activation=LeakyReLU(negative_slope=0.1), padding='same', kernel_regularizer=l2(0.02), name='D3_Conv2_Layer')(d3_concat)
    d3_se1 = SE_Layer(128, ratio=8, name='D3_SE1_Layer')(d3_conv2)
    d3_dropout = Dropout(0.2, name='D3_drop')(d3_se1)
    d3_conv3 = Conv3D(128, (3, 3, 3), activation=LeakyReLU(negative_slope=0.1), padding='same', kernel_regularizer=l2(0.02), name='D3_Conv3_Layer')(d3_dropout)

    #Decoder Block 2(D2)
    d2_up = UpSampling3D(size=(2, 2, 2), name='D2_up')(d3_conv3)
    d2_conv1 = Conv3D(64, (3, 3, 3), activation=LeakyReLU(negative_slope=0.1), padding='same', kernel_regularizer=l2(0.02), name='D2_Conv1_Layer')(d2_up)
    d2_instance = InstanceNormalization(name='D2_instance_Layer')(d2_conv1)
    d2_concat = concatenate([d2_instance, e2_dropout], name='D2_concat_Layer')
    d2_conv2 = Conv3D(64, (3, 3, 3), activation=LeakyReLU(negative_slope=0.1), padding='same', kernel_regularizer=l2(0.02), name='D2_Conv2_Layer')(d2_concat)
    d2_se1 = SE_Layer(64, ratio=8, name='D2_SE1_Layer')(d2_conv2)
    d2_dropout = Dropout(0.2, name='D2_drop')(d2_se1)
    d2_conv3 = Conv3D(64, (3, 3, 3), activation=LeakyReLU(negative_slope=0.1), padding='same', kernel_regularizer=l2(0.02), name='D2_Conv3_Layer')(d2_dropout)

    #Decoder Block 1(D1)
    d1_up = UpSampling3D(size=(2, 2, 2), name='D1_up')(d2_conv3)
    d1_conv1 = Conv3D(32, (3, 3, 3), activation=LeakyReLU(negative_slope=0.1), padding='same', name='D1_Conv1_Layer')(d1_up)
    d1_instance = InstanceNormalization(name='D1_instance_Layer')(d1_conv1)
    d1_concat = concatenate([d1_instance, e1_pc_embed], name='D1_concat_Layer')
    d1_conv2 = Conv3D(32, (3, 3, 3), activation=LeakyReLU(negative_slope=0.1), padding='same', name='D1_Conv2_Layer')(d1_concat)
    d1_se1 = SE_Layer(32, ratio=8, name='D1_SE1_Layer')(d1_conv2)
    d1_dropout = Dropout(0.2, name='D1_drop')(d1_se1)
    d1_conv3 = Conv3D(32, (3, 3, 3), activation=LeakyReLU(negative_slope=0.1), padding='same', name='D1_Conv3_Layer')(d1_dropout)

    #Probablity Filter
    prob = Conv3D(7, (1, 1, 1), activation='softmax', name='prob')(d1_conv3)

    output = prob

    model = Model(inputs=inputs, outputs=prob, name='MY_LATUP')
    model.compile(loss=loss, optimizer=Adam(beta_1=0.9, beta_2=0.999, learning_rate=0.0001), metrics=[MeanIoU(num_classes=7, name='IoU')])
    return model