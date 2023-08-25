import tensorflow as tf
from keras import layers
from keras.utils import plot_model

model_name = "ComNet"

"""
1 * _initial = 1 * Conv2D
(1 * _block = 2 * Conv2D)
3 * _stack = 2 * _block
1 * _head = 1 Dense

Total layers = 14
Total params: 2,782,154
"""


class ComNet:
    """ComNet"""

    def __init__(
        self,
    ):
        pass

    def _block(self, x, filters, stride=1, conv_short=True, name=None):
        """_block

        Args:
            x: input tensor.
            filters: integer, filters of Conv2D filter.
            short_conv: bool, channel matching Conv2D.
            index: integer, label of layer.
            name: str, block name.
        """
        if conv_short:
            shortcut = layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=stride,
                padding="same",
                use_bias=False,
                kernel_initializer="HeNormal",
                name=name + "_1x1conv_shortcut",
            )(x)
            shortcut = layers.BatchNormalization(name=name + "_bn_shortcut")(shortcut)
        else:
            shortcut = x

        # Conv2D_1
        x = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=stride,
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_3x3conv_1",
        )(x)
        x = layers.BatchNormalization(name=name + "_bn_1")(x)
        x = layers.Activation("relu", name=name + "_relu_1")(x)

        # Conv2D_2
        x = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_3x3conv_2",
        )(x)
        x = layers.BatchNormalization(name=name + "_bn_2")(x)

        # Residual_Connection
        x = layers.Add(name=name + "_residual_connection")([x, shortcut])
        x = layers.Activation("relu", name=name + "_relu_2")(x)

        return x

    def _stack(self, x, filters, stride, blocks, name):
        """_stack

        Args:
            x: input tensor
            filters: integer, filters of the Conv2D
            blocks: integer, blocks in the stack.
            name: string, stack name.
        """
        if stride > 1:
            flag = True
        else:
            flag = False

        x = self._block(
            x, filters, stride=stride, conv_short=flag, name=name + f"_block1"
        )

        for index in range(2, blocks + 1):
            x = self._block(
                x,
                filters,
                conv_short=False,
                name=name + f"_block{index}",
            )

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

        # Input
        input = layers.Input(shape=input_shape)
        x = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_network0" + "_3x3conv_0",
        )(input)
        x = layers.BatchNormalization(name=name + "_network0" + "_bn_0")(x)
        x = layers.Activation("relu", name=name + "_network0" + "_relu_0")(x)

        # 1
        x = self._stack(
            x=x,
            filters=filters,
            stride=1,
            blocks=2,
            name=name + "_Layer1",
        )

        # 2
        x = self._stack(
            x=x,
            filters=filters * 2,
            stride=2,
            blocks=2,
            name=name + "_Layer2",
        )

        # 3
        x = self._stack(
            x=x,
            filters=filters * 4,
            stride=2,
            blocks=2,
            name=name + "_Layer3",
        )
        output = self._head(x=x, classes=classes, name=name)

        # Model Build
        model = tf.keras.models.Model(
            inputs=input, outputs=output, name=f"{model_name}"
        )

        return model


if __name__ == "__main__":
    model = ComNet()._build(
        input_shape=(32, 32, 3), classes=10, filters=64, name="ComNet"
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
