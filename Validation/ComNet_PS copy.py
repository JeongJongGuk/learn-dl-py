import tensorflow as tf
from keras import layers
from keras.utils import plot_model

model_name = "ComNet_PS"

"""
1 * _initial = 1 * Conv2D
(1 * _block = 2 * Conv2D)
3 * _stack = 2 * _block
1 * _head = 1 Dense

Preactivation-ResNet
SENet

Total layers = 14
Total params: 
"""


class ComNet_PS:
    """ComNet_PS"""

    def __init__(
        self,
    ):
        pass

    def _se_block(self, x, ratio, name):
        """_se_block

        Args:
            x: input tensor.
            ratio: integer, reduction ratio.
            name: string, block name.
        """
        n_channel = int(x.shape[-1])

        se = layers.GlobalAveragePooling2D(name=name + "_avg_pool")(x)
        se = layers.Dense(n_channel * ratio, activation="relu", name=name + "_dense_1")(
            se
        )
        se = layers.Dense(n_channel, activation="sigmoid", name=name + "_dense_2")(se)
        x = layers.Multiply(name=name + "_mul")([x, se])

        return x

    def _block(self, x, filters, short_conv, name):
        """_block

        Args:
            x: input tensor.
            filters: integer, filters of Conv2D filter.
            short_conv: bool, channel matching Conv2D.
            index: integer, label of layer.
            name: string, block name.
        """
        preact = layers.BatchNormalization(name=name + "_bn_preact")(x)
        preact = layers.Activation("relu", name=name + "_relu_preact")(preact)

        if short_conv:
            shortcut = layers.Conv2D(
                filters=filters,
                kernel_size=1,
                strides=(1, 1),
                padding="same",
                use_bias=True,
                kernel_initializer="HeNormal",
                name=name + "_1x1conv_shortcut",
            )(x)
        else:
            shortcut = x

        # Conv2D_1
        x = layers.BatchNormalization(name=name + "_bn_1")(x)
        x = layers.Activation("relu", name=name + "_relu_1")(x)
        x = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_3x3conv_1",
        )(x)

        # Conv2D_2
        x = layers.BatchNormalization(name=name + "_bn_2")(x)
        x = layers.Activation("relu", name=name + "_relu_2")(x)
        x = layers.Conv2D(
            filters=filters,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            use_bias=True,
            kernel_initializer="HeNormal",
            name=name + "_3x3conv_2",
        )(x)

        # SE_Block
        x = self._se_block(x=x, ratio=16, name=name + "_se")

        # Residual_Connection
        x = layers.Add(name=name + "_residual_connection")([x, shortcut])

        return x

    def _stack(self, x, filters, blocks, short_conv, pool, name):
        """_stack

        Args:
            x: input tensor
            filters: integer, filters of the Conv2D
            blocks: integer, blocks in the stack.
            name: string, stack name.
        """
        for index in range(blocks):
            x = self._block(
                x, filters, short_conv=short_conv, name=name + f"_block{index+1}"
            )
            short_conv = False
        if pool:
            x = layers.MaxPool2D(pool_size=(2, 2), name=name + "_max_pool")(x)

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
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_network0" + "_3x3conv_0",
        )(input)
        x = self._stack(
            x=x,
            filters=filters * 2,
            blocks=2,
            short_conv=True,
            pool=True,
            name=name + "_Layer2",
        )
        x = self._stack(
            x=x,
            filters=filters * 4,
            blocks=2,
            short_conv=True,
            pool=False,
            name=name + "_Layer3",
        )
        output = self._head(x=x, classes=classes, name=name)

        # Model Build
        model = tf.keras.models.Model(
            inputs=input, outputs=output, name=f"{model_name}"
        )

        return model


if __name__ == "__main__":
    model = ComNet_PS()._build(
        input_shape=(32, 32, 3), classes=10, filters=64, name=model_name
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
