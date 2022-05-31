import os
import common.model as model
import common.datagen as datagen
from tensorflow.python.keras.callbacks import ModelCheckpoint, CSVLogger


print(os.environ.get('CUDA_HOME'))
print(os.environ.get('LD_LIBRARY_PATH'))



def train_model3D(train_files, val_files, in_weights_path, out_dir, BATCH, EPOCHS, IMG_SHAPE, model_type):
    aec3D = model.build_model(IMG_SHAPE, model_type)
    aec3D.summary()

    if in_weights_path is not None:
        aec3D.load_weights(in_weights_path)

    if val_files is None:
        N = int(len(train_files) * 0.9)
        val_files = train_files[N:]
        train_files = train_files[:N]

    print('training on', len(train_files))
    print('validating on', len(val_files))

    # Generators
    training_generator = datagen.Data3DGeneratorAEC(train_files, augment=False, batch_size=BATCH, dim=IMG_SHAPE)
    validation_generator = datagen.Data3DGeneratorAEC(val_files, augment=False, batch_size=BATCH, dim=IMG_SHAPE)

    log_fname = os.path.join(out_dir, 'log.csv')

    checkpoint_path = os.path.join(out_dir, "model.{epoch:02d}-{val_loss:.2f}.hdf5")
    callbacks = [ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=True, mode='auto'),
                  CSVLogger(log_fname, append=True)]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Train model on dataset
    aec3D.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=EPOCHS,
                        callbacks=callbacks,
                        workers=-1,
                        use_multiprocessing=True)
    model_checkpoint_path = os.path.join(out_dir, 'trained_%d_epochs' % EPOCHS)
    aec3D.save_weights(model_checkpoint_path)
    return model_checkpoint_path


def train_model(train_files, val_files, in_weights_path, out_dir, BATCH, EPOCHS, IMG_SHAPE, model_type):
    aec2D = model.build_model(IMG_SHAPE, model_type)
    aec2D.summary()

    if in_weights_path is not None:
        aec2D.load_weights(in_weights_path)

    if val_files is None:
        N = int(len(train_files) * 0.9)
        val_files = train_files[N:]
        train_files = train_files[:N]

    print('training on', len(train_files))
    print('validating on', len(val_files))

    # Generators
    training_generator = datagen.Data2DGeneratorAEC(train_files, augment=False, batch_size=BATCH, dim=IMG_SHAPE)
    validation_generator = datagen.Data2DGeneratorAEC(val_files, augment=False, batch_size=BATCH, dim=IMG_SHAPE)

    log_fname = os.path.join(out_dir, 'log.csv')

    checkpoint_path = os.path.join(out_dir, "model.{epoch:02d}-{val_loss:.2f}.hdf5")
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor='val_acc', verbose=0, save_best_only=False, save_weights_only=True,
                        mode='auto'),
        CSVLogger(log_fname, append=True)]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    # Train model on dataset
    aec2D.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=EPOCHS,
                        callbacks=callbacks,
                        workers=-1,
                        use_multiprocessing=True)
    model_checkpoint_path = os.path.join(out_dir, 'trained_%d_epochs' % EPOCHS)
    aec2D.save_weights(model_checkpoint_path)

    return model_checkpoint_path
