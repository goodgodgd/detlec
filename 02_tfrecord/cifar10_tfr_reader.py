import os.path as op
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Config:
    RAW_DATA_PATH = "/home/ian/workspace/detlec/dataset/cifar-10-batches-py"
    TFRECORD_PATH = "/home/ian/workspace/detlec/dataset/tfrecords"
    CLASS_NAMES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    CIFAR_IMG_SHAPE = (32, 32, 3)
    BATCH_SIZE = 32


def read_cifar10_tfrecord():
    gpu_config()
    train_dataset = get_dataset(Config.TFRECORD_PATH, "cifar10", "train", True, Config.BATCH_SIZE)
    test_dataset = get_dataset(Config.TFRECORD_PATH, "cifar10", "test", False, Config.BATCH_SIZE)
    check_data(train_dataset)
    classifier = AdvancedClassifier(Config.BATCH_SIZE)
    classifier.build_model(Config.CIFAR_IMG_SHAPE, len(Config.CLASS_NAMES))
    classifier.train(train_dataset, test_dataset, 5)
    classifier.evaluate(test_dataset)


def get_dataset(tfr_path, dataname, split, shuffle=False, batch_size=32, epochs=1):
    tfr_files = tf.io.gfile.glob(op.join(tfr_path, f"{dataname}_{split}", "*.tfrecord"))
    tfr_files.sort()
    print("[TfrecordReader] tfr files:", tfr_files)
    dataset = tf.data.TFRecordDataset(tfr_files)
    dataset = dataset.map(parse_example)
    dataset = set_properties(dataset, shuffle, epochs, batch_size)
    return dataset


def parse_example(example):
    features = {
        "image": tf.io.FixedLenFeature(shape=(), dtype=tf.string, default_value=""),
        "label_index": tf.io.FixedLenFeature(shape=(), dtype=tf.int64, default_value=0),
        "label_name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    }
    parsed = tf.io.parse_single_example(example, features)
    # method 1. decode from raw bytes
    parsed["image"] = tf.io.decode_raw(parsed["image"], tf.uint8)
    parsed["image"] = tf.reshape(parsed["image"], Config.CIFAR_IMG_SHAPE)
    parsed["image_u8"] = parsed["image"]    # only for visualize
    parsed["image"] = tf.image.convert_image_dtype(parsed["image"], dtype=tf.float32)   # for model input
    # method 2. decode from png format
    # parsed["image"] = tf.io.decode_png(parsed["image"])
    return parsed


def set_properties(dataset, shuffle, epochs, batch_size):
    if shuffle:
        dataset = dataset.shuffle(100)
    dataset = dataset.batch(batch_size).repeat(epochs)
    return dataset


def check_data(dataset):
    for i, features in enumerate(dataset):
        print("sample", i, features["image_u8"].shape, features["image"].shape, features["label_index"].numpy(), features["label_name"].numpy())
        if i == 0:
            show_samples(features["image_u8"], features["label_name"])
        if i > 10:
            break


def show_samples(images, labels, grid=(3, 3)):
    plt.figure(figsize=grid)
    num_samples = grid[0] * grid[1]
    for i in range(num_samples):
        plt.subplot(grid[0], grid[1], i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i].numpy())
        plt.xlabel(labels[i].numpy().decode())
    plt.show()



"""
Copied from tf_classifier_adv.py
"""
from tensorflow import keras
from tensorflow.keras import layers
from timeit import default_timer as timer


class DurationTime:
    def __init__(self, context):
        self.start = 0
        self.context = context

    def __enter__(self):        # entering 'with' context
        self.start = timer()
        return self             # pass object by 'as'

    def __exit__(self, type, value, trace_back):    # exiting 'with' context
        print(f"{self.context}: {timer() - self.start:1.2f}")


def gpu_config():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


class AdvancedClassifier:
    def __init__(self, batch_size=32):
        self.model = keras.Model()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.RMSprop()
        self.batch_size = batch_size

    def build_model(self, input_shape, output_shape):
        input_tensor = layers.Input(shape=input_shape)
        x = layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", name="conv1")(input_tensor)
        x = layers.MaxPool2D(pool_size=(2, 2), name="pooling1")(x)
        x = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", name="conv2")(x)
        x = layers.MaxPool2D(pool_size=(2, 2), name="pooling2")(x)
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(units=100, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.2)(x)
        output_tensor = layers.Dense(units=output_shape, activation="softmax", name="dense2")(x)
        self.model = keras.Model(inputs=input_tensor, outputs=output_tensor, name="tf-classifier")
        self.model.summary()
        keras.utils.plot_model(self.model, "tf-clsf-model-adv.png")

    def train(self, train_dataset, val_dataset, epochs):
        with DurationTime("** training time"):
            for epoch in range(epochs):
                for i, features in enumerate(train_dataset):
                    self.train_batch_graph(features["image"], features["label_index"])
                loss, accuracy = self.evaluate(val_dataset)
                print(f"[Training] epoch={epoch}, val_loss={loss:1.4f}, val_accuracy={accuracy:1.4f}")

    @tf.function
    def train_batch_graph(self, x_batch, y_batch):
        """
        :param x_batch: input image [B, H, W, 3], float
        :param y_batch: label indices [B], int
        """
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(x_batch, training=True)
            loss = self.loss_object(y_batch, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def evaluate(self, dataset, verbose=True):
        y_true, y_pred = [], []
        for features in dataset:
            probs = self.model(features["image"])
            y_pred.append(probs)
            y_true.append(features["label_index"])
        y_pred = tf.concat(y_pred, axis=0).numpy()
        y_true = tf.concat(y_true, axis=0).numpy()
        accuracy = np.mean(np.argmax(y_pred, axis=1) == y_true)
        loss = self.loss_object(y_true, y_pred).numpy()
        if verbose:
            print("=== evaluate result ===")
            np.set_printoptions(precision=4, suppress=True)
            print("  prediction shape:", y_pred.shape, y_true.shape)
            print("  pred indices", np.argmax(y_pred, axis=1)[:20])
            print("  true indices", y_true[:20])
            print(f"  loss={loss:1.4f}, accuracy={accuracy:1.4f}")
        return loss, accuracy


if __name__ == "__main__":
    read_cifar10_tfrecord()
