from keras.models import Model
from keras.layers import Input, PReLU, Dense, LSTM, multiply, concatenate, Activation
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout, MaxPooling1D

from utils.constants import MAX_SEQUENCE_LENGTH_LIST, NB_CLASSES_LIST
from utils.keras_utils import train_model, evaluate_model, set_trainable, visualize_context_vector, visualize_cam
from utils.layer_utils import AttentionLSTM

DATASET_INDEX = 87
MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH_LIST[DATASET_INDEX]
NB_CLASS = NB_CLASSES_LIST[DATASET_INDEX]

TRAINABLE = True


def generate_model():
    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = LSTM(128)(ip)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    #model.load_weights("weights/beef_weights - 8667 v3 lstm 8 batch 128 no attention dropout 80.h5")

    return model


def generate_model_2():
    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    x = AttentionLSTM(128)(ip)
    x = Dropout(0.8)(x)

    y = Permute((2, 1))(ip)
    y = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    return model

def generate_model_3():
    ip = Input(shape=(1, MAX_SEQUENCE_LENGTH))

    # x = LSTM(64, return_sequences=True)(ip)
    # x = LSTM(64, return_sequences=True)(x)
    x = LSTM(64)(ip)
    x = Dropout(0.8)(x)   

    out = Dense(NB_CLASS, activation='softmax')(x)
    
    model = Model(ip, out)

    model.summary()

    # add load model code here to fine-tune

    return model


if __name__ == "__main__":
    model = generate_model()

    # train_model(model, DATASET_INDEX, dataset_prefix='fmri_fcn_zscore_spatial_b64', epochs=2000, batch_size=64,normalize_timeseries=False)

    evaluate_model(model, DATASET_INDEX+1, dataset_prefix='fmri_fcn_zscore_spatial_b64', batch_size=64, normalize_timeseries=False)

    # visualize_context_vector(model, DATASET_INDEX, dataset_prefix='beef', visualize_sequence=True,
    #                         visualize_classwise=True, limit=1)

    # visualize_cam(model, DATASET_INDEX, dataset_prefix='beef', class_id=0)
