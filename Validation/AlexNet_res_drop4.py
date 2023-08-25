import tensorflow as tf
from keras import layers
from keras.utils import plot_model

model_name = "AlexNet_res_drop4"

"""
skip_connection: 2 * Conv2D
conv_pool: 4 * skip_connection
model: 3 * conv_pool 

layers: 1 + 2 * 4 * 3 + 1 = 26 layers
"""


class AlexNet_res_drop4:
    """AlexNet_res_drop"""

    def __init__(
        self,
    ):
        pass

    def skip_connection_sub(self, x, num_filter, conv_dropout_rate, name, number):
        """skip_connection_sub
        Args:
            x: Input tensor
            num_filter: Number of Conv2D filter
            name: str
            number: str
        """

        # Conv2D_Skip
        x1 = layers.Conv2D(
            filters=num_filter,
            kernel_size=3,
            strides=(2, 2),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + f"_3x3_conv_skip_{number}_1",
        )(x)
        x1 = layers.BatchNormalization(name=name + f"_bn_skip_{number}_1")(x1)
        x1 = layers.Activation("relu", name=name + f"_relu_skip_{number}_1")(x1)

        x1 = layers.Conv2D(
            filters=num_filter,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + f"_3x3_conv_skip_{number}_2",
        )(x1)
        x1 = layers.BatchNormalization(name=name + f"_bn_skip_{number}_2")(x1)

        # Residual_Connection
        x = layers.Conv2D(
            filters=num_filter,
            kernel_size=1,
            strides=(2, 2),
            padding="same",
            use_bias=True,
            kernel_initializer="HeNormal",
            name=name + "_1x1_conv_skip_matching",
        )(x)
        x = layers.Add(name=name + f"residual_connection_{number}")([x, x1])
        x = layers.Activation("relu", name=name + f"_relu_skip_{number}")(x)

        x = layers.Dropout(conv_dropout_rate)(x)

        return x

    def skip_connection(self, x, num_filter, conv_dropout_rate, name, number):
        """skip_connection
        Args:
            x: Input tensor
            num_filter: Number of Conv2D filter
            name: str
            number: str
        """

        # Conv2D_Skip
        x1 = layers.Conv2D(
            filters=num_filter,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + f"_3x3_conv_skip_{number}_1",
        )(x)
        x1 = layers.BatchNormalization(name=name + f"_bn_skip_{number}_1")(x1)
        x1 = layers.Activation("relu", name=name + f"_relu_skip_{number}_1")(x1)

        x1 = layers.Conv2D(
            filters=num_filter,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + f"_3x3_conv_skip_{number}_2",
        )(x1)
        x1 = layers.BatchNormalization(name=name + f"_bn_skip_{number}_2")(x1)

        # Residual_Connection
        x = layers.Add(name=name + f"residual_connection_{number}")([x, x1])
        x = layers.Activation("relu", name=name + f"_relu_skip_{number}")(x)

        x = layers.Dropout(conv_dropout_rate)(x)

        return x

    def conv_pool_1(self, x, num_filter, conv_dropout_rate, name):
        """conv_pool_1
        Input Shape: 32x32
        Output Shape: 32x32
        """

        # Conv2D_1
        x = layers.Conv2D(
            filters=num_filter,
            kernel_size=3,
            strides=(1, 1),
            padding="same",
            use_bias=False,
            kernel_initializer="HeNormal",
            name=name + "_3x3_conv",
        )(x)
        x = layers.BatchNormalization(name=name + "_bn")(x)
        x = layers.Activation("relu", name=name + "_relu")(x)

        # Skip_Connection
        x = self.skip_connection(x, num_filter, conv_dropout_rate, name, 1)
        x = self.skip_connection(x, num_filter, conv_dropout_rate, name, 2)
        x = self.skip_connection(x, num_filter, conv_dropout_rate, name, 3)
        x = self.skip_connection(x, num_filter, conv_dropout_rate, name, 4)

        return x

    def conv_pool_2(self, x, num_filter, conv_dropout_rate, name):
        """conv_pool_2
        Input Shape: 32x32
        Output Shape: 16x16
        """

        # Skip_Connection
        x = self.skip_connection_sub(x, num_filter, conv_dropout_rate, name, 1)
        x = self.skip_connection(x, num_filter, conv_dropout_rate, name, 2)
        x = self.skip_connection(x, num_filter, conv_dropout_rate, name, 3)
        x = self.skip_connection(x, num_filter, conv_dropout_rate, name, 4)

        return x

    def conv_pool_3(self, x, num_filter, conv_dropout_rate, name):
        """conv_pool_3
        Input Shape: 16x16
        Output Shape: 8x8
        """

        # Skip_Connection
        x = self.skip_connection_sub(x, num_filter, conv_dropout_rate, name, 1)
        x = self.skip_connection(x, num_filter, conv_dropout_rate, name, 2)
        x = self.skip_connection(x, num_filter, conv_dropout_rate, name, 3)
        x = self.skip_connection(x, num_filter, conv_dropout_rate, name, 4)

        return x

    def classifier(self, x, num_class, dropout_rate, name):
        """
        Flatten -> Dropout -> Dense -> Dropout -> Dense -> Softmax

        Args:
            x: input
            output_num: 출력 이미지 개수
        """

        # GlobalAveragePolling2D
        x = layers.GlobalAveragePooling2D(name=name + "gloavgpool")(x)
        x = layers.Dropout(dropout_rate)(x)

        # Dense
        x = layers.Dense(
            units=num_class,
            use_bias=True,
            kernel_initializer="HeNormal",
            name=name + "_dense",
        )(x)
        x = layers.Softmax(name=name + "_softmax")(x)

        return x

    def _build(
        self,
        input_shape,
        num_class,
        num_filter,
        dropout_rate,
        conv_dropout_rate,
    ):
        """
        AlexNet Build

        Args:
         num_filter: Conv2D 필터 개수 (Pool 통과시 마다 2배 씩 증가하게 만듬.)
         input_shape: 입력 데이터 형상
         output_num: 출력 데이터 개수
        """

        # Input
        input = layers.Input(shape=input_shape)

        # Block 1
        x = self.conv_pool_1(
            x=input,
            num_filter=num_filter,
            conv_dropout_rate=conv_dropout_rate,
            name="ResNet_block_1",
        )

        # Block 2
        x = self.conv_pool_2(
            x=x,
            num_filter=num_filter * 2,
            conv_dropout_rate=conv_dropout_rate,
            name="ResNet_block_2",
        )

        # Block 3
        x = self.conv_pool_3(
            x=x,
            num_filter=num_filter * 4,
            conv_dropout_rate=conv_dropout_rate,
            name="ResNet_block_3",
        )

        # Output
        output = self.classifier(x, num_class, dropout_rate, name="classifier")

        # Model Build
        model = tf.keras.models.Model(
            inputs=input, outputs=output, name=f"{model_name}"
        )

        return model


if __name__ == "__main__":
    model = AlexNet_res_drop4()._build(
        input_shape=(32, 32, 3),
        num_class=10,
        num_filter=64,
        dropout_rate=0.3,
        conv_dropout_rate=0.2,
    )

    # 모델 시각화 그래프 생성 후 이미지 파일로 저장
    plot_model(
        model,
        to_file=f"model_image/{model_name}.png",
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True,
    )
