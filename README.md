Tensorflow 2.13
CUDA 11.8
cuDNN 8.6

AlexNet
1. Activataion
    - ReLU: Dying ReLU 문제.
    - GELU: Dying ReLU 문제 해결, but Exponential 연산으로 인한 Computation Resource 요구량 상승

2. Head
    - Flatten: Dense Layer로 인한 Params 증가
    - GlobalAveragePooling: Dense Layer로 인한 params 감소

3. BatchNormalization
    - with BN: Deep Layer Model 학습 수렴 성공률 상승
    - without BN: Deep Layer Model 학습 수렴 실패

4. Residual Connection
    - Deep Layer Model 학습 수렴 가능하게 만듬.

DenseNet
1. ResNet과 달리 Channel을 쌓는 방식(기존 Filter Params 재활용)으로 학습.

ResNet
1. Residual Connection을 통해 
    - Vanishing Gradient 문제 해결. -> 학습 수렴 성공률 높임.

Validation
Plain**: 비교 대상

1. ComNet - ResNet 참고

2. ComNet_P - Preact-ResNet 참고

3. ComNet_PS - Preact-ResNet, SENet 참고