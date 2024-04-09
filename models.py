from tensorflow.keras.layers import Flatten, Reshape, Conv1D, MaxPooling1D, Conv1DTranspose, Dense, UpSampling1D
from keras.models import Sequential

def AD1(input_shape = (20, 9), z_size = 3):
    model = Sequential()
    input_shape = input_shape

    model.add(Conv1D(32, 3, activation='relu', strides=1, padding="same", input_shape= input_shape))
    model.add(Conv1D(32, 3, activation='relu', strides=1, padding="same"))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(64, 3, activation='relu', strides=1, padding="same"))
    model.add(Conv1D(64, 3, activation='relu', strides=1, padding="same"))
    model.add(MaxPooling1D(2))

    model.add(Conv1D(64, 3, activation='relu', strides=1, padding="same"))
    model.add(Conv1D(64, 3, activation='relu', strides=1, padding="same"))
    model.add(MaxPooling1D(2))


    # model.add(Dropout(0.25))
    model.add(Conv1D(1, 1, activation='relu', strides=1, padding="same"))
    model.add(Flatten())
    model.add(Dense(z_size, activation='relu', strides=1, padding="same"))
    
    model.add(Dense(input_shape[0]//2//2//2, activation='relu', strides=1, padding="same"))
    
    model.add(Conv1DTranspose(64, 3, activation='relu', strides=1, padding="same"))
    model.add(Conv1DTranspose(64, 3, activation='relu', strides=1, padding="same"))
    model.add(UpSampling1D(2))
    
    model.add(Conv1DTranspose(64, 3, activation='relu', strides=1, padding="same"))
    model.add(Conv1DTranspose(64, 3, activation='relu', strides=1, padding="same"))
    model.add(UpSampling1D(2))
    
    model.add(Conv1DTranspose(32, 3, activation='relu', strides=1, padding="same", input_shape= input_shape))
    model.add(Conv1DTranspose(32, 3, activation='relu', strides=1, padding="same"))
    model.add(UpSampling1D(2))
    
    model.add(Conv1D(1, 3, activation='sigmoid', strides=1, padding="same"))
    
    return model