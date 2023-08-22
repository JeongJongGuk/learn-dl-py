import tensorflow as tf


def split_states(x, n):
    *start, m = x.shape.as_list()
    return tf.reshape(x, start + [n, m // n])


def merge_states(x):
    *start, a, b = x.shape.as_list()
    return tf.reshape(x, start + [a * b])


# 입력 텐서 정의
input_shape = (None, 10, 60)
input_tensor = tf.keras.layers.Input(shape=input_shape[1:])  # 배치 크기를 제외한 나머지 차원


# shape_list 함수 정의
def shape_list(x):
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


# input_tensor의 형태 확인
input_shape = shape_list(input_tensor)
print("Input shape:", input_shape)  # 출력: [None, 10, 60]

n = 2
# split_states 함수 사용
split_tensor = split_states(input_tensor, n)
# split_tensor의 형태 확인
split_shape = shape_list(split_tensor)
print("Split shape:", split_shape)  # 출력: [None, 10, 2, 30]

# merge_states 함수 사용
merged_tensor = merge_states(split_tensor)
# merged_tensor의 형태 확인
merged_shape = shape_list(merged_tensor)
print("Merged shape:", merged_shape)  # 출력: [None, 10, 60]
