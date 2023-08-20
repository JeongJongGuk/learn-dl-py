import tensorflow as tf


def load_cifar10():
    # Cifar-10 Load
    (train_images, train_labels), (
        test_images,
        test_labels,
    ) = tf.keras.datasets.cifar10.load_data()

    # Normalization and One-Hot Encoding
    train_images = train_images / 255.0
    test_images = test_images / 255.0
    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    return train_images, train_labels, test_images, test_labels


# 데이터 증강 함수 정의
def augment_image(image, mode):
    # 여러 개의 증강 기법 정의
    augmentations = [
        lambda x: tf.image.random_flip_left_right(x),
        lambda x: tf.image.random_brightness(x, max_delta=0.2),
        lambda x: tf.image.random_contrast(x, lower=0.5, upper=1.0),
        lambda x: tf.image.random_hue(x, max_delta=0.05),
        lambda x: tf.image.random_saturation(x, lower=0.8, upper=1.0),
    ]

    # 여러 개의 증강 기법 중 무작위로 선택한 인덱스
    selected_index = tf.random.shuffle(tf.range(len(augmentations)))[0]

    # 선택된 증강 기법을 적용
    augmentation = augmentations[selected_index]
    if mode:
        image = augmentation(image)
        # print(selected_index)

    return image
