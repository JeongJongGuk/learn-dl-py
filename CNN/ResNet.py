import tensorflow as tf
from keras.utils import plot_model
from keras import layers


class ResNet:
    """ResNet110 Cifar-10"""

    def __init__(
        self,
    ):
        pass

    def residual_block_1(self, x, num_filter, dropout_rate, name):
        """Residual connection"""

        # Conv2D_1
        x1 = layers.Conv2D(
            filters=num_filter,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_3x3_conv_1",
        )(x)
        if dropout_rate:
            x1 = layers.Dropout(rate=dropout_rate, name=name + "_dropout_1")(x1)
        x1 = layers.BatchNormalization(name=name + "_bn_1")(x1)
        x1 = layers.Activation("relu", name=name + "_relu_1")(x1)

        # Conv2D_2
        x1 = layers.Conv2D(
            filters=num_filter,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_3x3_conv_2",
        )(x1)
        if dropout_rate:
            x1 = layers.Dropout(rate=dropout_rate, name=name + "_dropout_2")(x1)
        x1 = layers.BatchNormalization(name=name + "_bn_2")(x1)
        x1 = layers.Activation("relu", name=name + "_relu_2")(x1)

        # Residual Connection
        x = layers.Add(name=name + "residual_connection")([x, x1])

        return x

    def residual_block_2(self, x, num_filter, dropout_rate, name):
        """Residual connection with Decrease the input shape about half"""

        # Conv2D_0
        x1 = layers.Conv2D(
            filters=num_filter,
            kernel_size=1,
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_skip_1x1_conv",
        )(x)
        if dropout_rate:
            x1 = layers.Dropout(rate=dropout_rate, name=name + "_skip_dropout")(x1)
        x1 = layers.BatchNormalization(name=name + "_skip_bn")(x1)
        x1 = layers.Activation("relu", name=name + "_skip_relu")(x1)

        # Conv2D_1
        x2 = layers.Conv2D(
            filters=num_filter,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_3x3_conv_1",
        )(x)
        if dropout_rate:
            x2 = layers.Dropout(rate=dropout_rate, name=name + "_dropout_1")(x2)
        x2 = layers.BatchNormalization(name=name + "_bn_1")(x2)
        x1 = layers.Activation("relu", name=name + "_relu_1")(x1)

        # Conv2D_2
        x2 = layers.Conv2D(
            filters=num_filter,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_3x3_conv_2",
        )(x2)
        if dropout_rate:
            x2 = layers.Dropout(rate=dropout_rate, name=name + "_dropout_2")(x2)
        x2 = layers.BatchNormalization(name=name + "_bn_2")(x2)
        x2 = layers.Activation("relu", name=name + "_relu_2")(x2)

        # Residual Connection
        x = layers.Add(name=name + "residual_connection")([x1, x2])

        return x

    def block_1(self, x, num_filter, dropout_rate, name):
        x = layers.Conv2D(
            filters=num_filter,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "3x3_conv",
        )(x)
        x = layers.BatchNormalization(name="bn")(x)
        x = layers.Activation("relu", name=name + "_relu")(x)
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_1"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_2"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_3"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_4"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_5"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_6"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_7"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_8"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_9"
        )

        return x

    def block_2(self, x, num_filter, dropout_rate, name):
        x = self.residual_block_2(
            x, num_filter, dropout_rate, name=name + "_residual_block_1"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_2"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_3"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_4"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_5"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_6"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_7"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_8"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_9"
        )
        return x

    def block_3(self, x, num_filter, dropout_rate, name):
        x = self.residual_block_2(
            x, num_filter, dropout_rate, name=name + "_residual_block_1"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_2"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_3"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_4"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_5"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_6"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_7"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_8"
        )
        x = self.residual_block_1(
            x, num_filter, dropout_rate, name=name + "_residual_block_9"
        )
        return x

    def classifier(self, x, num_class, name):
        """
        GAP -> Dense

        Args:
         x: input
         output_num: 출력 이미지 개수
        """
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(
            units=num_class, use_bias=False, name=name + "_dense"
        )(x)
        x = tf.keras.layers.Softmax(name=name + "_softmax")(x)

        return x

    def _build(self, input_shape, num_class, num_filter, dropout_rate):
        """
        ResNet18 Build

        Args:
         num_filter: Conv2D 필터 개수
         input_shape: 입력 데이터 형상
         output_num: 출력 데이터 개수
        """
        # Input
        input = tf.keras.layers.Input(shape=input_shape)

        # Blocks
        x = self.block_1(input, num_filter, dropout_rate, name="ResNet_Block_1")
        x = self.block_2(x, num_filter * 2, dropout_rate, name="ResNet_Block_2")
        x = self.block_3(x, num_filter * 4, dropout_rate, name="ResNet_Block_3")

        # Output
        output = self.classifier(x, num_class, name="ResNet_Classifier")

        # Model Build
        model = tf.keras.models.Model(inputs=input, outputs=output)

        return model


if __name__ == "__main__":
    model = ResNet()
    model._build(
        input_shape=(32, 32, 3),
        num_class=10,
        num_filter=16,
        dropout_rate=0.2,
    )

    # 모델 시각화 그래프 생성 후 이미지 파일로 저장
    plot_model(model, to_file="ResNet.png", show_shapes=True, show_layer_names=True)
