import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pprint
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


def load_dataset(dataname="cifar10"):
    if dataname == "cifar10":
        dataset = tf.keras.datasets.cifar10
    else:
        raise ValueError(f"Invalid dataset name: {dataname}")

    (x_train, y_train), (x_test, y_test) = dataset.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    print(f"Load {dataname} dataset:", x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return (x_train, y_train), (x_test, y_test)


"""
Classifier
"""
def tf2_keras_classifier():
    gpu_config()
    train_data, test_data = load_dataset()
    model = create_model(train_data)
    train_model(model, train_data)
    test_model(model, test_data)


def create_model(dataset, use_add=True):
    x, y = dataset
    input_shape = x.shape[1:]
    num_class = tf.reduce_max(y).numpy() + 1
    print(f"[create_model] input shape={input_shape}, num_class={num_class}")

    if use_add:
        model = keras.Sequential(name="tf-classifier")
        model.add(layers.Input(shape=input_shape))
        model.add(layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", name="conv1"))
        model.add(layers.MaxPool2D(pool_size=(2, 2), name="pooling1"))
        model.add(layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", name="conv2"))
        model.add(layers.MaxPool2D(pool_size=(2, 2), name="pooling2"))
        model.add(layers.Flatten(name="flatten"))
        model.add(layers.Dense(units=100, activation="relu", name="dense1"))
        model.add(keras.layers.Dropout(0.2))
        model.add(layers.Dense(units=num_class, activation="softmax", name="dense2"))
    else:
        model = keras.Sequential([
            layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu", input_shape=input_shape, name="conv1"),
            layers.MaxPool2D(pool_size=(2, 2), name="pooling1"),
            layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu", name="conv2"),
            layers.MaxPool2D(pool_size=(2, 2), name="pooling2"),
            layers.Flatten(name="flatten"),
            layers.Dense(units=100, activation="relu", name="dense1"),
            keras.layers.Dropout(0.2),
            layers.Dense(units=num_class, activation="softmax", name="dense2"),
            ],
            name="tf-classifier")
    model.summary()
    keras.utils.plot_model(model, "tf-clsf-model.png")
    return model


def train_model(model, train_data, split_ratio=0.8):
    x, y = train_data
    trainlen = int(tf.shape(x)[0].numpy() * split_ratio)
    x_train, y_train = x[:trainlen], y[:trainlen]
    x_val, y_val = x[trainlen:], y[trainlen:]

    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer=keras.optimizers.RMSprop(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    with DurationTime("** training time") as duration:
        history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_val, y_val))
    history = {key: np.array(values) for key, values in history.history.items()}
    np.set_printoptions(precision=4, suppress=True)
    pp = pprint.PrettyPrinter(indent=2, width=100, compact=True)
    print("[train_model] training history:")
    pp.pprint(history)


def test_model(model, test_data):
    x_test, y_test = test_data
    print("[test_model] evaluate by model.evaluate()")
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"  test loss: {loss:1.4f}")
    print(f"  test accuracy: {accuracy:1.4f}")
    print("[test_model] predict by model.predict()")
    predicts = model.predict(x_test)
    print("  prediction shape:", predicts.shape, y_test.shape)
    print("  first 5 predicts:\n", predicts[:5])
    print("  check probability:", np.sum(predicts[:5], axis=1))
    print("  manual accuracy:", np.mean(np.argmax(predicts, axis=1) == y_test[:, 0]))


if __name__ == "__main__":
    tf2_keras_classifier()
