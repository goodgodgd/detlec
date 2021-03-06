import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer


"""
Common utils
"""
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


def load_dataset(dataname="cifar10", show_imgs=True):
    if dataname == "cifar10":
        dataset = tf.keras.datasets.cifar10
        class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    else:
        raise ValueError(f"Invalid dataset name: {dataname}")

    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = y_train[:, 0], y_test[:, 0]
    print(f"Load {dataname} dataset:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    if show_imgs:
        show_samples(x_train, y_train, class_names)
    return (x_train, y_train), (x_test, y_test)


def show_samples(images, labels, class_names, grid=(3,4)):
    plt.figure(figsize=grid)
    num_samples = grid[0] * grid[1]
    for i in range(num_samples):
        plt.subplot(grid[0], grid[1], i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.xlabel(class_names[labels[i]])
    plt.show()


"""
Classifier
"""
class AdvancedClassifier:
    def __init__(self, batch_size=32, val_ratio=0.2):
        self.model = keras.Model()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.RMSprop()
        self.batch_size = batch_size
        self.val_ratio = val_ratio

    def build_model(self, x, y):
        input_shape = x.shape[1:]
        num_class = tf.reduce_max(y).numpy() + 1
        input_tensor = layers.Input(shape=input_shape)
        x = layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", name="conv1")(input_tensor)
        x = layers.MaxPool2D(pool_size=(2, 2), name="pooling1")(x)
        x = layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", name="conv2")(x)
        x = layers.MaxPool2D(pool_size=(2, 2), name="pooling2")(x)
        x = layers.Flatten(name="flatten")(x)
        x = layers.Dense(units=100, activation="relu", name="dense1")(x)
        x = layers.Dropout(0.2)(x)
        output_tensor = layers.Dense(units=num_class, activation="softmax", name="dense2")(x)
        self.model = keras.Model(inputs=input_tensor, outputs=output_tensor, name="tf-classifier")
        self.model.summary()
        keras.utils.plot_model(self.model, "tf-clsf-model-adv.png")

    def train(self, x, y, epochs):
        """
        :param x: input image [N, H, W, 3], float
        :param y: label indices [N], int
        :param epochs: # epochs to train
        """
        trainlen = int(x.shape[0] * (1 - self.val_ratio))
        x_train, y_train = x[:trainlen], y[:trainlen]
        x_val, y_val = x[trainlen:], y[trainlen:]

        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        dataset = dataset.shuffle(200).batch(self.batch_size)
        with DurationTime("** training time") as duration:
            for epoch in range(epochs):
                for x_batch, y_batch in dataset:
                    self.train_batch_graph(x_batch, y_batch)
                loss, accuracy = self.evaluate(x_val, y_val, verbose=False)
                print(f"[Training] epoch={epoch}, val_loss={loss}, val_accuracy={accuracy}")

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

    def evaluate(self, x, y_true, verbose=True):
        if verbose:
            print("[evaluate] predict by model.__call__()")
        y_pred = self.model(x)
        accuracy = np.mean(np.argmax(y_pred, axis=1) == y_true)
        loss = self.loss_object(y_true, y_pred)
        if verbose:
            np.set_printoptions(precision=4, suppress=True)
            print("  prediction shape:", y_pred.shape, y_true.shape)
            print("  first 5 predicts:\n", y_pred[:5].numpy())
            print("  check probability:", np.sum(y_pred[:5], axis=1))
            print(f"  loss={loss:1.4f}, accuracy={accuracy:1.4f}")
        return loss, accuracy


def tf2_advanced_classifier():
    gpu_config()
    (x_train, y_train), (x_test, y_test) = load_dataset("cifar10")
    clsf = AdvancedClassifier()
    clsf.build_model(x_train, y_train)
    clsf.train(x_train, y_train, 5)
    clsf.evaluate(x_test, y_test)


if __name__ == "__main__":
    tf2_advanced_classifier()
