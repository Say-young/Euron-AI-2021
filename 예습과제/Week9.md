## ✔ CS231n 9주차 예습과제
(세미나 수강 후 내용 보완하여 블로그에 정리 글 올릴 예정)

1. CNN Architectures

main architecture : AlexNet, VGG, GoogLeNet, RestNet

* AlexNet
    * Conv - Max Pool - Norm 반복 구조

* VGGNet
    * 7x7 conv 대신 3x3conv 2개 사용
    * parameter의 개수 filter 크기에 increase by square
    * 파라미터 개수를 줄이려면 FC를 최소한으로 이용 or 없애야 > 이것을 적용한 게 GoogLeNet
    * deeper, smaller total spatial area를 이용하여 컴퓨팅 코스트를 일정하게 만들었다는 것
    * VGG19보다 VGG16을 더 많이 사용

* GoogleNet
    * Inception module은 Network안에 Network 구조를 차용
    * 다양한 크기의 filter를 이용함으로써 good local network topology를 형성
    * conv가 parallel하게 있으므로 우리는 병렬적으로 이를 실행할 수 있으므로 computing efficiency가 매우 뛰어남
    * 단점
        * pooling layer도 depth를 증가시킴
        * input channel과 output channel의 크기가 커서 computational complexity가 큼
        > bottleneck layer(1x1 conv)를 통해 original input의 dimension을 1x1 conv layer의 depth로 projection함. 이를 통해서 computational complexity를 관리할 수 있게 됨.
    * 요약하자면 더 깊어졌고, 더 높은 효율성을 가지며, FC Layer가 없다.

* ResNet
    * residual connection을 이용해 수많은 block을 쌓아올린 것
    * input과 output이 direct하게 mapping되지 않고, identity를 더한 어떤 잔차를 학습시키는 형태로 네트워크가 진행
    * GoogLeNet과 마찬가지로 computation efficiency를 위해 bottleneck layer를 추가할 수 있음.