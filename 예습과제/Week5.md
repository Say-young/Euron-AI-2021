예습 과제를 블로그에 정리하고 있습니다. https://say-young.tistory.com/entry/CS231n-Lec5?category=923651

## CS231n Convolutional Neural Networks for Visual Recognition

1. ConvNet (CNN/Convolutional Neural Networks)

ConvNet은 우리가 이전까지 살펴봤던 일반 신경망과 매우 유사하다. 학습 가능한 가중치(weight)와 편향(bias)으로 구성되어 있다. 각 뉴런은 입력을 받아 내적 연산(dot product)을 한 뒤, 선택에 따라 non-linear 연산을 한다. 전체 네트워크는 일반 신경망과 마찬가지로 미분 가능한 하나의 score function을 갖게 된다. 마지막 레이어에 loss function을 가지며, 우리가 일반 신경망을 학습시킬 때 사용하던 각종 기법들을 동일하게 적용할 수 있다.

ConvNet 아키텍처는 입력 데이터가 image라는 가정을 하기 대문에 image data가 갖는 특성들을 인코딩할 수 있다. 이러한 아키텍처는 forward function을 더욱 효과적으로 구현할 수 있고, 네트워크를 학습시키는데 필요한 parameter의 수를 크게 줄일 수 있게 해준다.

ConvNet이 사용되는 예시
image retrieval(really powerful for doing similarity matching), detection(localiziging where in an image is, a bus or a boat, and so on, and raw precise bounding boxes around that.), face-recognition(get out a likehood of who this person is.), pose-recognition, interpretation medical image, classification for galaxy, image captioning 등등 굉장히 다양한 분야에서 사용되고 있다.
일반 신경망은 이미지를 다루기에 적절하지 않다. CIFAR-10 데이터의 경우 각 이미지가 32x32x3 (가로,세로 32, 3개 컬러 채널)로 이뤄져 있어서 첫 번째 히든 레이어 내의 하나의 뉴런의 경우 32x32x3=3072개의 가중치가 필요하지만, 더 큰 이미지를 사용할 경우에는 같은 구조를 이용하는 것이 불가능하다. 게다가 이런 뉴런이 레이어 내에 여러개 존재하므로 모수의 개수가 크게 증가하게 된다. 이와 같이 Fully-connectivity는 심한 낭비이며 많은 수의 모수는 곧 오버피팅(overfitting)을 야기한다.

일반 신경망과 달리, ConvNet의 레이어들은 가로,세로,깊이의 3개 차원을 갖게 된다 (여기에서 말하는 깊이란 전체 신경망의 깊이가 아니라 액티베이션 볼륨 (activation volume) 에서의 3번 째 차원을 의미한다.). 예를 들어 CIFAR-10 이미지는 32x32x3 (가로,세로,깊이) 의 차원을 갖는 입력 액티베이션 볼륨 (activation volume)이라고 볼 수 있다. 하나의 레이어에 위치한 뉴런들은 일반 신경망과는 달리 앞 레이어의 전체 뉴런이 아닌 일부에만 연결이 되어 있다. ConvNet 아키텍쳐는 전체 이미지를 클래스 점수들로 이뤄진 하나의 벡터로 만들어주기 때문에 마지막 출력 레이어는 1x1x10(10은 CIFAR-10 데이터의 클래스 개수)의 차원을 가지게 된다.

즉, ConvNet은 여러 레이어로 이루어져 있다. 각각의 레이어는 3차원의 볼륨을 입력으로 받고 미분 가능한 함수를 거쳐 3차원의 볼륨을 출력하는 기능을 한다. ConvNet의 각 레이어는 미분 가능한 변환 함수를 통해 하나의 activation volume을 또 다른 activation volume으로 변환시킨다. ConvNet 아키텍쳐에서는 크게 convolutional layer, pooling layer, fully-connected layer라는 3개의 레이어가 사용된다.



2. Convolution Layer

Conv layer의 출력은 3차원으로 정렬된 뉴런들로 해석될 수 있다. 기존에는 왼쪽 이미지와 같이 input image를 길게 늘여서 계산했기 때문에 공간 정보가 유지되지 않았으며 이렇게 계산된 값은 하나의 특징만을 나타내게 되었다. (공간 정보를 포함하지 못한다.)

하지만 CNN(오른쪽)에서는 Conv layer를 도입해 공간 정보를 유지한 채 convolution(합성곱)을 진행한다. 이미지보다 작은 filter를 정의한 뒤 이미지의 좌측 상단부터 차례대로 dot product를 진행한다. dot product를 통해 나온 하나의 값은 activation map의 하나의 값을 구성한다. 이 filter가 옆으로 정해진 칸 수만큼 이동하면서 차례대로 dot product를 진행하여 activation map을 완성시키는 것이 Conv layer다.

filter의 개수만큼 activation map이 형성되고 Conv layer를 쌓아나가면서 CNN을 완성시킨다. CIFAR-10 데이터를 다루기 위한 간단한 ConvNet은 [INPUT-CONV-RELU-POOL-FC]로 구축할 수 있다.

CNN은 어떻게 공간 정보를 포함한 채로 합성곱을 진행할 수 있을까? filter가 한번 dot product를 진행할 때 filter가 속한 공간에 있는 특징을 추출하여 값을 내보내고, 이러한 방식으로 이미지의 모든 구역을 훑고 난 뒤 생성된 activation map은 각 공간에 대한 특징값들의 모임으로 생각하면 된다. 따라서 이미지 전체의 특징을 추출하는 것이 아닌 공간별 특징을 추출하여 표현하기 때문에 공간 정보를 유지한다고 하는 것이다. CNN에서는 3개의 hyperparameter가 출력 볼륨의 크기를 결정한다. 바로 깊이, stride, zero-padding이다.

어떤 간격(가로/세로의 공간적 간격)으로 깊이 컬럼을 할당할 지를 의미하는 stride를 결정해야 한다. filter가 합성곱을 진행할 때 '몇 칸씩 움직일 지'에 해당하는 값이 stride다.

stride 값이 1인 경우이다. stride만큼 필터가 칸을 옮겨가며 합성곱이 진행되는 것을 볼 수 있다. 주로 Conv Layer에는 stride 1을 사용하는데, 보통 작은 stride가 더 잘 동작할 뿐만 아니라 모든 spatial downsampling을 Pool layer에 맡기게 되고, Conv Layer는 입력 볼륨의 깊이만 변화시키게 된다.

그렇다면 stride를 3으로 설정한다면? 적용할 수 없다. 이렇게 stride에 따라 output size는 오른쪽의 공식에 따라 계산할 수 있다. stride 값이 커질수록 output size는 작아진다.

합성곱을 진행하고 나면 output size가 input size보다 작아지는 문제가 발생한다. 따라서 이를 방지하기 위해서 input image를 0으로 둘러싼 뒤에 합성곱을 진행하게 된다. (테두리를 채운 후 합성곱을 진행해보자. output size가 input size와 동일하게 나오는 것을 확인할 수 있다.) Conv Layer를 통과하면서 spatial 크기를 그대로 유지하게 해준다는 점 이외에도, 패딩을 쓰면 성능도 향상된다. 만약 제로 패딩을 하지 않고 valid convolution을 한다면 볼륨의 크기는 Conv Layer를 거칠 때마다 줄어들게 되고, 가장자리의 정보들이 빠르게 사라진다.



3. Pooling Layer

위에서 소개했듯이, CNN은 Conv layer, RELU, Pooling layer가 한 세트로 이루어져 있으며, 이것들을 차례로 쌓은 후, 마지막을 fully-connected layer로 구성한다. pooling layer는 spatial 차원에 대한 downsampling을 위해 사용된다. 추출한 특징을 최대한 유지한 채 이미지의 크기를 줄여주는 장치이다.

가장 많이 사용되는 방식은 Max Pooling 방식이다. (별다른 언급이 없다면 Max Pool이라고 가정한다.) Pooling layer에서 사용할 filter의 크기와 stride 값을 정한 뒤 filter가 합성될 때 이전처럼 dot product를 진행하는 것이 아니라 max 값만 산출하는 것이다. 이렇게 하면 가장 큰 특징을 살리면서 이미지의 크기를 줄여나갈 수 있다.

이후 fully-connected layer에 넣어 각 class 별로 예측치를 산출하면 CNN이 완성된다. fully-connected 레이어 내의 뉴런들은 일반 신경망 챕터에서 보았듯이 이전 레이어의 모든 activation들과 연결되어 있다. 그러므로 fully-connected 레이어의 액티베이션은 합성곱을 한 뒤 bias를 더해 구할 수 있다.