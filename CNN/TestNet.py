import tensorflow as tf
from keras.utils import plot_model
from keras import layers


class TestNet:
    def __init__(self) -> None:
        pass

    def conv_pool_1(self, x, num_filter, name):
        # Conv2D
        x = layers.Conv2D(
            filters=num_filter,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_3x3_conv_1",
        )(x)
        x = layers.Activation("relu", name=name + "_relu")(x)
        x = layers.Conv2D(
            filters=num_filter,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_3x3_conv_2",
        )(x)
        x = layers.Activation("relu", name=name + "_relu")(x)
        x = layers.Conv2D(
            filters=num_filter,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_3x3_conv_3",
        )(x)
        x = layers.Activation("relu", name=name + "_relu")(x)

        # MaxPool2D
        x = layers.MaxPooling2D((2, 2), name=name + "_pool")(x)

        return x
