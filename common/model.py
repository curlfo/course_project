
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Conv2D, MaxPooling2D,\
        UpSampling2D, Flatten, Reshape, Input
from tensorflow.keras.optimizers import Adam


def build_model(shape, model_type=0):
    if model_type == 0:
        return build_model_1(shape)
    elif model_type == 1:
        return build_model_2(shape)
    elif model_type == 2:
        return build_model_3(shape)
    elif model_type == 3:
        return build_model_4(shape)
    elif model_type == 4:
        return build_model_5(shape)
    elif model_type == 5:
        return build_model_6(shape)

def show_model_summary(IMG_SHAPE = (256, 256, 1), model_type=0):
    aec = build_model(IMG_SHAPE, model_type)
    aec.summary()

def build_model_1(shape):
    autoencoder = Sequential()

    # Encoder Layers
    autoencoder.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same', input_shape=shape))
    autoencoder.add(MaxPooling3D((2, 2, 2), padding='same'))
    autoencoder.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling3D((2, 2, 2), padding='same'))
    autoencoder.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling3D((2, 2, 2), padding='same'))
    autoencoder.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling3D((2, 2, 2), padding='same'))
    autoencoder.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling3D((2, 2, 2), padding='same'))
    autoencoder.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling3D((2, 2, 2), padding='same'))

    # Flatten encoding for visualization
    autoencoder.add(Flatten(name='encoded'))

    autoencoder.add(Reshape((int(shape[0]/64), int(shape[1]/64), int(shape[2]/64), 32)))

    # Decoder Layers
    autoencoder.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling3D((2, 2, 2)))
    autoencoder.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling3D((2, 2, 2)))
    autoencoder.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling3D((2, 2, 2)))
    autoencoder.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling3D((2, 2, 2)))
    autoencoder.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling3D((2, 2, 2)))
    autoencoder.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling3D((2, 2, 2)))
    autoencoder.add(Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same'))

    opt = Adam(lr=1e-4)
    autoencoder.compile(optimizer=opt, loss='binary_crossentropy')
    #autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder


def build_model_2(shape):

    autoencoder = Sequential()

    # Encoder Layers
    autoencoder.add(Conv3D(16, (3, 3, 3), activation='relu', padding='same', input_shape=shape))
    autoencoder.add(MaxPooling3D((2, 2, 2), padding='same'))
    autoencoder.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling3D((2, 2, 2), padding='same'))
    autoencoder.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling3D((2, 2, 2), padding='same'))
    autoencoder.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling3D((2, 2, 2), padding='same'))
    autoencoder.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling3D((2, 2, 2), padding='same'))
    autoencoder.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling3D((2, 2, 2), padding='same'))
    autoencoder.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling3D((2, 2, 2), padding='same'))

    # Flatten encoding for visualization
    autoencoder.add(Flatten(name='encoded'))

    autoencoder.add(Reshape((int(shape[0]/128), int(shape[1]/128), int(shape[2]/128), 128)))

    # Decoder Layers
    autoencoder.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling3D((2, 2, 2)))
    autoencoder.add(Conv3D(128, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling3D((2, 2, 2)))
    autoencoder.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling3D((2, 2, 2)))
    autoencoder.add(Conv3D(64, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling3D((2, 2, 2)))
    autoencoder.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling3D((2, 2, 2)))
    autoencoder.add(Conv3D(32, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling3D((2, 2, 2)))
    autoencoder.add(Conv3D(16, (3, 3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling3D((2, 2, 2)))
    autoencoder.add(Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same'))

    opt = Adam(lr=1e-4)
    autoencoder.compile(optimizer=opt, loss='binary_crossentropy')
    #autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder


def build_model_3(shape):
    autoencoder = Sequential()
    autoencoder.add(Input(shape=shape))
    autoencoder.add(Conv2D(128, (5, 5), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(1, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', name='encoded'))

 # Flatten encoding for visualization
    #autoencoder.add(Flatten(name='encoded'))
    #autoencoder.add(Reshape((int(shape[0]/128), int(shape[1]/128), int(shape[2]/128), 128)))

    autoencoder.add(Conv2D(1, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(128, (3, 3), activation='sigmoid', padding='same', name='output'))

    opt = Adam(lr=1e-4)
    autoencoder.compile(optimizer=opt, loss='binary_crossentropy')

    return autoencoder

def build_model_4(shape):
    autoencoder = Sequential()
    autoencoder.add(Input(shape=shape))
    autoencoder.add(Conv2D(16, (5, 5), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', name='encoded'))

 # Flatten encoding for visualization
    #autoencoder.add(Flatten(name='encoded'))
    #autoencoder.add(Reshape((int(shape[0]/128), int(shape[1]/128), int(shape[2]/128), 128)))

    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='output'))

    opt = Adam(lr=1e-4)
    autoencoder.compile(optimizer=opt, loss='binary_crossentropy')

    return autoencoder

def build_model_5(shape):
    autoencoder = Sequential()
    autoencoder.add(Input(shape=shape))
    autoencoder.add(Conv2D(16, (5, 5), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', name='encoded'))

 # Flatten encoding for visualization
    #autoencoder.add(Flatten(name='encoded'))
    #autoencoder.add(Reshape((int(shape[0]/128), int(shape[1]/128), int(shape[2]/128), 128)))

    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='output'))

    opt = Adam(lr=1e-4)
    autoencoder.compile(optimizer=opt, loss='binary_crossentropy')

    return autoencoder

def build_model_6(shape):
    autoencoder = Sequential()
    autoencoder.add(Input(shape=shape))
    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same'))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(MaxPooling2D((2, 2), padding='same', name='encoded'))

    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    autoencoder.add(UpSampling2D((2, 2)))
    autoencoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same', name='output'))

    opt = Adam(lr=1e-4)
    autoencoder.compile(optimizer=opt, loss='binary_crossentropy')

    return autoencoder