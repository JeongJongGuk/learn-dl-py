import tensorflow as tf
from keras.utils import plot_model
from keras import layers


class DenseNet:
    """DenseNet

    1. BottleNeck Layer
    2. Dense Block
    3. Transition Block
    """

    def __init__(self):
        pass

    # 1x1 3x3 Conv2D Block
    def bottleneck_layer(self, x, growth_rate, dropout_rate, name):
        x1 = layers.BatchNormalization(name=name + "_bn_1")(x)
        x1 = layers.Activation("relu", name=name + "_relu_1")(x1)
        x1 = layers.Conv2D(
            growth_rate * 4,
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_1x1_conv",
        )(x1)
        if dropout_rate:
            x1 = layers.Dropout(rate=dropout_rate, name=name + "_dropout_1")(x1)
        x1 = layers.BatchNormalization(name=name + "_bn_2")(x1)
        x1 = layers.Activation("relu", name=name + "_relu_2")(x1)
        x1 = layers.Conv2D(
            growth_rate,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_3x3_conv",
        )(x1)
        if dropout_rate:
            x1 = layers.Dropout(rate=dropout_rate, name=name + "_dropout_2")(x1)
        x = layers.concatenate(inputs=[x, x1], name=name + "_concat")
        return x

    # Dense Block
    def dense_block(self, x, num_layer, growth_rate, dropout_rate, name):
        for i in range(num_layer):
            x = self.bottleneck_layer(
                x,
                growth_rate,
                dropout_rate,
                name=name + "_bottleneck_layer_" + str(i + 1),
            )
        return x

    # Transition Block
    def transition_block(self, x, compression, dropout_rate, name):
        x = layers.BatchNormalization(name=name + "_bn_1")(x)
        x = layers.Activation("relu", name=name + "_relu_1")(x)
        x = layers.Conv2D(
            int(x.shape[-1] * compression),
            kernel_size=1,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_1x1_conv",
        )(x)
        if dropout_rate:
            x = layers.Dropout(rate=dropout_rate, name=name + "_dropout")(x)
        x = layers.BatchNormalization(name=name + "_bn_2")(x)
        x = layers.Activation("relu", name=name + "_relu_2")(x)
        x = layers.AveragePooling2D((2, 2), name=name + "_pool")(x)
        return x

    # Build Model
    def _build(self, input_shape, num_class, growth_rate, dropout_rate):
        """
        Args:
         input_shape: tensor; input_shape.
         num_class: integer; truth label.
         growh_rate: float; growing layer number in dense block.
         dropout_rate: float; dropout_rate after Conv2D
        """
        # Input
        input = layers.Input(shape=input_shape)
        x = layers.Conv2D(
            growth_rate * 2,
            kernel_size=3,
            strides=1,
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name="conv1",
        )(input)
        # Dense Blocks
        x = self.dense_block(x, 16, growth_rate, dropout_rate, name="dense_1")
        x = self.transition_block(x, 0.5, dropout_rate, name="pool_1")
        x = self.dense_block(x, 16, growth_rate, dropout_rate, name="dense_2")
        x = self.transition_block(x, 0.5, dropout_rate, name="pool_2")
        x = self.dense_block(x, 16, growth_rate, dropout_rate, name="dense_3")
        # Output
        x = layers.BatchNormalization(name="bn")(x)
        x = layers.ReLU(name="relu")(x)
        x = layers.GlobalAveragePooling2D(name="globalavg_pool")(x)
        output = tf.keras.layers.Dense(
            num_class, activation="softmax", name="classifier"
        )(x)
        # Model Build
        model = tf.keras.models.Model(inputs=input, outputs=output, name="DenseNet")

        return model


if __name__ == "__main__":
    model = DenseNet()
    model._build(
        input_shape=(32, 32, 3),
        num_class=10,
        growth_rate=12,
        dropout_rate=0.2,
    )

    # 모델 시각화 그래프 생성 후 이미지 파일로 저장
    plot_model(model, to_file="DenseNet.png", show_shapes=True, show_layer_names=True)
