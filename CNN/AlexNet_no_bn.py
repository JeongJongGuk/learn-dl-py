import tensorflow as tf
from keras import layers
from keras.utils import plot_model


class AlexNet:
    """AlexNet"""

    def __init__(
        self,
    ):
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
            name=name + "_3x3_conv",
        )(x)
        # x = layers.BatchNormalization(name=name + "_bn")(x)
        x = layers.Activation("relu", name=name + "_relu")(x)

        # MaxPool2D
        x = layers.MaxPooling2D((2, 2), name=name + "_pool")(x)

        return x

    def conv_pool_2(self, x, num_filter, name):
        # Conv2D
        x = layers.Conv2D(
            filters=num_filter,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_3x3_conv",
        )(x)
        # x = layers.BatchNormalization(name=name + "_bn")(x)
        x = layers.Activation("relu", name=name + "_relu")(x)

        # MaxPool2D
        x = layers.MaxPool2D((2, 2), name=name + "_pool")(x)

        return x

    def conv_pool_3(self, x, num_filter, name):
        # Conv2D_1
        x = layers.Conv2D(
            filters=num_filter,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_3x3_conv_1",
        )(x)
        # x = layers.BatchNormalization(name=name + "_bn_1")(x)
        x = layers.Activation("relu", name=name + "_relu_1")(x)

        # Conv2D_2
        x = layers.Conv2D(
            filters=num_filter,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_3x3_conv_2",
        )(x)
        # x = layers.BatchNormalization(name=name + "_bn_2")(x)
        x = layers.Activation("relu", name=name + "_relu_2")(x)

        # Conv2D_3
        x = layers.Conv2D(
            filters=num_filter,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_3x3_conv_3",
        )(x)
        # x = layers.BatchNormalization(name=name + "_bn_3")(x)
        x = layers.Activation("relu", name=name + "_relu_3")(x)

        # MaxPool2D
        x = layers.MaxPool2D((2, 2), name=name + "_pool")(x)

        return x

    def classifier(self, x, num_class, dropout_rate, compression, name):
        """
        Flatten -> Dropout -> Dense -> Dropout -> Dense -> Softmax

        Args:
            x: input
            output_num: 출력 이미지 개수
            dropout_rate: Dropout 비율
            compresssion: Dense Unit 압축률
        """

        # Flatten
        x = layers.Flatten()(x)
        dense_unit = x.shape[-1] * compression

        # Dense_1
        if dropout_rate:
            x = layers.Dropout(rate=dropout_rate, name="dropout_1")(x)
        x = layers.Dense(
            units=dense_unit,
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_dense_1",
        )(x)
        # x = layers.BatchNormalization(name=name + "_bn_1")(x)
        x = layers.Activation("relu", name=name + "_relu_1")(x)

        # Dense_2
        if dropout_rate:
            x = layers.Dropout(rate=dropout_rate, name="dropout_2")(x)
        x = layers.Dense(
            units=dense_unit / 2,
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_dense_2",
        )(x)
        # x = layers.BatchNormalization(name=name + "_bn_2")(x)
        x = layers.Activation("relu", name=name + "_relu_2")(x)

        # Dense_3
        x = layers.Dense(
            units=num_class,
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_dense_3",
        )(x)
        x = layers.Softmax(name=name + "_softmax")(x)

        return x

    def _build(self, input_shape, num_class, num_filter, dropout_rate, compression):
        """
        AlexNet Build

        Args:
         num_filter: Conv2D 필터 개수 (Pool 통과시 마다 2배 씩 증가하게 만듬.)
         input_shape: 입력 데이터 형상
         output_num: 출력 데이터 개수
         dropout_rate: Classifier dropout_rate 조절
         compression: Classifier compression 조절
        """

        # Input
        input = layers.Input(shape=input_shape)

        # Block 1
        x = self.conv_pool_1(x=input, num_filter=num_filter, name="block1")

        # Block 2
        x = self.conv_pool_2(x=x, num_filter=num_filter * 2, name="block2")

        # Block 3
        x = self.conv_pool_3(x=x, num_filter=num_filter * 4, name="block3")

        # Output
        output = self.classifier(
            x, num_class, dropout_rate, compression, name="classifier"
        )

        # Model Build
        model = tf.keras.models.Model(inputs=input, outputs=output, name="AlexNet")

        return model


if __name__ == "__main__":
    model = AlexNet()._build(
        input_shape=(32, 32, 3),
        num_class=10,
        num_filter=64,
        dropout_rate=0.5,
        compression=0.25,
    )

    # 모델 시각화 그래프 생성 후 이미지 파일로 저장
    plot_model(
        model,
        to_file="model_image/AlexNet_no_bn.png",
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True,
        show_trainable=True,
    )
