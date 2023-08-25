import tensorflow as tf
from keras import layers
from keras.utils import plot_model

model_name = "Plain14"

"""
1 * _initial = 1 * Conv2D
3 * _stack = 4 * Conv2D
1 * _head = 1 Dense

Total layers = 14
Total params: 2,739,658
"""


class Plain14:
    """Plain14"""

    def __init__(self) -> None:
        pass

    def _block(self, x, filters, index, name):
        """_block

        Args:
            x: input tensor.
            filters: integer, filters of the Conv2D.
            name: string, stack name.
        """
        x = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + f"_3x3conv_{index}",
        )(x)
        x = layers.BatchNormalization(name=name + f"_bn_{index}")(x)
        x = layers.Activation("relu", name=name + f"_relu_{index}")(x)

        return x

    def _stack(self, x, filters, blocks, name):
        """_stack

        Args:
            x: input tensor.
            filters: integer, filters of the Conv2D.
            blocks: integer, blocks in the stack.
            name: string, stack name.
        """
        for index in range(blocks):
            x = self._block(x, filters, index=index + 1, name=name)
        x = layers.MaxPool2D(pool_size=(2, 2), name=name + "max_pool")(x)

        return x

    def _head(self, x, classes, name):
        """_head

        Args:
            x: input tensor.
            classes: integer, number of output.
            name: string, head name.
        """
        x = layers.GlobalAveragePooling2D(name=name + "_avg_pool")(x)
        x = layers.Dense(
            units=classes, activation="softmax", name=name + "_predictions"
        )(x)

        return x

    def _build(self, input_shape, classes, filters, name):
        """_build

        Args:
            x: input tensor.
            classes: integer, number of output.
            name: string, model name.
        """
        input = layers.Input(shape=input_shape)
        x = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_3x3conv_0",
        )(input)
        x = layers.BatchNormalization(name=name + "_Layer0" + "_bn_0")(x)
        x = layers.Activation("relu", name=name + "_Layer0" + "relu_0")(x)
        x = self._stack(x=x, filters=filters, blocks=4, name=name + "_Layer1")
        x = self._stack(x=x, filters=filters * 2, blocks=4, name=name + "_Layer2")
        x = self._stack(x=x, filters=filters * 4, blocks=4, name=name + "_Layer3")
        output = self._head(x=x, classes=classes, name=name)

        model = tf.keras.models.Model(inputs=input, outputs=output)

        return model


if __name__ == "__main__":
    model = Plain14()._build(
        input_shape=(32, 32, 3), classes=10, filters=64, name="Plain14"
    )
    model.summary()

    # 모델 시각화 그래프 생성 후 이미지 파일로 저장
    plot_model(
        model,
        to_file=f"model_image/{model_name}.png",
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True,
    )
