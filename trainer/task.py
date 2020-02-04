# network structure is based on
# https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5

from . import util
import tensorflow as tf
import numpy as np
import os
from matplotlib import pyplot as plt
import argparse

# create neccessary folders
def init(job_dir):
    tf.gfile.MkDir(os.path.join(job_dir, 'data'))
    tf.gfile.MkDir(os.path.join(job_dir, 'snapshot'))
    tf.gfile.MkDir(os.path.join(job_dir, 'keras_tensorboard'))


# based on Google Cloud's example
def get_args():
    """Argument parser.

    Returns:
        Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting models')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=20,
        help='number of times to go through the data, default=20')
    parser.add_argument(
        '--batch-size',
        default=128,
        type=int,
        help='number of records to read during each training step, default=128')
    parser.add_argument(
        '--learning-rate',
        default=.01,
        type=float,
        help='learning rate for gradient descent, default=.01')
    parser.add_argument(
        '--verbosity',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default='INFO')
    args, _ = parser.parse_known_args()
    return args


# Google Cloud dataset generating function
def input_fn(features, labels, shuffle, num_epochs, batch_size):
    """Generates an input function to be used for model training.

    Args:
    features: numpy array of features used for training or inference
    labels: numpy array of labels for each example
    shuffle: boolean for whether to shuffle the data or not (set True for
        training, False for evaluation)
    num_epochs: number of epochs to provide the data for
    batch_size: batch size for training

    Returns:
    A tf.data.Dataset that can provide data to the Keras model for training or
        evaluation
    """
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(features))

    # We call repeat after shuffling, rather than before, to prevent separate
    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset


def create_model(input_dim, output_dim):
    Dense = tf.keras.layers.Dense
    LSTM = tf.keras.layers.LSTM
    Dropout = tf.keras.layers.Dropout
    Activation = tf.keras.layers.Activation

    model = tf.keras.Sequential([
        LSTM(512, input_shape=input_dim, return_sequences=True),
        Dropout(0.3),
        LSTM(512, return_sequences=True),
        Dropout(0.3),
        LSTM(512),
        Dense(256),
        Dropout(0.3),
        Dense(output_dim),
        Activation("softmax")
    ])

    return model


# implementation based on GCloud's example
def visualize(history):
    # Visualize History for Loss.
    plt.title('Keras model loss')
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.show()

    # Visualize History for Accuracy.
    plt.title('Keras model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.legend(['training', 'validation'], loc='lower right')
    plt.show()


# a keras callback to clean checkpoint models and upload to cloud 
class UploadCheckPointCallback(tf.keras.callbacks.Callback):
    def __init__(self, job_dir):
        self.job_dir = job_dir

    def on_epoch_end(self, epoch, logs=None):
        dest_dir = os.path.join(self.job_dir, 'snapshot')

        for file in tf.gfile.Glob('*.hdf5'):
            file = os.path.basename(file)
            dest_file = os.path.join(dest_dir, file)

            print('Try to copy {} to {}'.format(file, dest_file))
            tf.gfile.Copy(file, dest_file, overwrite=True)


def train(args):
    # load dataset
    dataset = util.load_data(args.job_dir)

    # split dataset into training and validation dataset
    split_dataset = util.split(dataset)
    train_x = split_dataset["train_x"]
    train_y = split_dataset["train_y"]
    val_x = split_dataset["val_x"]
    val_y = split_dataset["val_y"]

    # create input generators
    training_dataset = input_fn(
        features=train_x,
        labels=train_y,
        shuffle=True,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size
    )

    validation_dataset = input_fn(
        features=val_x,
        labels=val_y,
        shuffle=True,
        num_epochs=args.num_epochs,
        batch_size=val_x.shape[0]
    )

    input_dim = (train_x.shape[1], train_x.shape[2])
    output_dim = (train_y.shape[1])

    # load saved model and resume training
    file = tf.gfile.Glob(os.path.join(args.job_dir, '*.hdf5'))
    source_file = file[0]
    dest_file = './{}'.format(os.path.basename(source_file))
    print(tf.gfile.ListDirectory(args.job_dir))
    print(source_file, dest_file)
    if args.job_dir != '.':
        tf.gfile.Copy(source_file, dest_file, overwrite=True)
    model = tf.keras.models.load_model(dest_file)

    # create model
    # model = create_model(input_dim, output_dim)``
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.summary()

    # Setup Checkpoint callback
    filepath = 'model-{epoch:02d}-{loss:.4f}.hdf5'
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath, monitor='loss', verbose=0, save_best_only=True, mode='min'
    )

    tensorboard_cb = tf.keras.callbacks.TensorBoard(
      os.path.join(args.job_dir, 'keras_tensorboard'),
      histogram_freq=1)

    earlystopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, verbose=1)

    history = model.fit(training_dataset,
                        epochs=args.num_epochs,
                        steps_per_epoch=int(train_x.shape[0] / args.batch_size),
                        validation_data=validation_dataset,
                        validation_steps=1,
                        callbacks=[checkpoint_cb, tensorboard_cb, earlystopping_cb,
                            UploadCheckPointCallback(args.job_dir)],
                        verbose=1)

    return history


if __name__ == "__main__":
    # tf.debugging.set_log_device_placement(True)

    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    args = get_args()
    init(args.job_dir)
    history = train(args)

    # Visualize model after training
    # visualize(history)