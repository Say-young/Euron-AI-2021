## ✔ CS231n 2주차 예습과제

(5:00~)

### 1. 😺 Image Classification pipeline
* 이미지 분류는 컴퓨터 비전의 가장 메인 task 중 하나
    * input image -> 라벨링 되어 있는 이미지군들을 사용해 들어온 이미지를 인지(카테고리에 할당)
    * 이미지가 800x600 픽셀이면 각 픽셀은 RGB를 표현하기 위한 3자리의 수로 이루어져 있음 (= 매우 많은 양의 수들!)
    * 이 수들로 라벨링된 데이터에 따라 이 이미지가 어떤 군에 속하는지 분류하는 것은 사실 매우 어려운 일이다. (물체가 바뀌는 것은 물론, 대상이 같아도 카메라가 움직이거나 빛의 위치가 달라지면 픽셀값이 달라진다.)

* sementic gap
    * 컴퓨터는 이미지를 숫자로 인식한다

* How?
    1. Attempts have been made
        * find edges(compute the edges of image) -> try to categorize all the different corners and boundaries...
        * 다루기 힘들고, 이 방대한 작업을 카테고리마다 전부 실행해야 함.
    2. Data-Driven Approach
        1. 인터넷에서 매우 많은 양의 데이터셋(이미지)를 모은다
        2. 데이터들을 학습을 통해 구분하고 정리해서 모델(이러한 이미지 카테고리들을 어떻게 인식/분류하는지에 대해 학습한 지식을 요약한 것)을 만든다
        3. 이 모델을 사용해 새로운 이미지를 인식

* train
    * input images and labels output a model
* predict
    * input the model and make predictions for images

***

#### 세미나 필기 추가

* 규칙 기반 판단의 문제점
    * view point(사진 찍은 각도)
    * illumination (밝기)
    * deformation
    * occulusion
    * background clutter (물체가 배경과 매우 흡사한 경우)
    * interclass variation (여러 종이 한 이미지에)

* 하지만 객체 하나하나를 모두 구분해내는 규칙을 모두 사람이 정의하는 것은 어렵다.
    * 그래서 나온 data driven approach
        * 안정성 : image input이 다양하게 주어지기 때문에 일정 부분 불안정성 해소 가능
        * 확장성
    * data set(labellig 된 이미지 수집) -> train(classifier을 학습시킴) -> evaluate(새로운 이미지를 input으로 주고 제대로 작동하는지 판단)

***


### 2. 🖥 first classifier: Nearest Neighbor
* 학습된 이미지를 모두 기억하고
* 새로운 이미지를 받아서 학습된 데이터들 중 가장 비슷한 이미지를 찾는다.

---

* Manhattan distance to compare images
    * 각각의 픽셀을 비교(tst image - training image)해서 모든 칸의 값을 더하는 방법
    * CIFAR10 : 머신러닝에 주로 사용되는 데이터 셋

    ---

    * memorize training data
        ![image](https://user-images.githubusercontent.com/55133871/112739477-9a8e5500-8faf-11eb-9988-327cdbe7bc5c.png)

    * L1 함수를 이용해 training set 중 가장 비슷한 것을 찾는 과정
        ![image](https://user-images.githubusercontent.com/55133871/112739492-abd76180-8faf-11eb-986c-a87722eaa1d2.png)

    * Q) 이 예시를 사용하면 학습과 예측에 걸리는 시간은?
    * A) 학습에는 O(1), 예측에는 O(N)
        * 모든 데이터를 비교해 예측하는 과정에 걸리는 시간이 길다. 학습에 걸리는 시간은 길어도 되지만 예측은 빨리 되어야 한다. 다양한 디바이스로 접근하는 사용자들은 빠르게 이미지가 분류되길 원할테니까.

    ---

    * decision reasons
        ![image](https://user-images.githubusercontent.com/55133871/112739620-c4944700-8fb0-11eb-922e-d4524f6a4c21.png)
        * (K=1) 초록색 area에 노란색 점이 있고, 파란색 영역으로 삐져나간 초록색 area가 있고 등등의 점으로 인해 이것은 noise가 낀 좋지 않은 상태라는 것을 알 수 있다.
        * K-Nearest Neighbors
            * k 값을 조정해 smooth한 결과값을 낸다
            * white region : there was no majority among the k-nearest neighbors

---

* K-Nearest Neighbors(kNN / lazy model) : Distance Metric
    * 새로운 데이터가 주어졌을 때 기존 데이터 중 가장 가까운 k개 이웃의 정보를 기반으로 새로운 데이터를 예측하는 방법 (모델을 별도로 구축하지 않고 관측치만을 이용)
    * k=1은 단 1개의 이웃만 보고 예측하는 것 // k가 너무 작을 경우 overfitting, 너무 클 경우 underfitting
    * 이미지를 분류하는 작업 이외에도 문장 유사도 등 다양한 것을 분류하는데 이 알고리즘을 확장해 사용할 수 있다.

    ![image](https://user-images.githubusercontent.com/55133871/112739818-72542580-8fb2-11eb-93df-6ee20bbb1676.png)
    * L1(Manhattan) distance
        * 두 점의 차의 절댓값(거리) 나타냄. 좌표계에 따라 달라짐
    * L2(Euclidean) distance
        * 두 점 사이의 '직선 거리' 나타냄. 좌표계의 영향을 받지 않음

    * Why distance?
        * 거리는 일종의 유사도(similarity)와 같은 개념
        * 거리가 가까울 수록 특성(feature)들이 비슷하다는 뜻

    ---

    * robust
        * 머신러닝에서 일반화(generalization)는 일부 특정 데이터만 잘 설명하는(=overfitting) 것이 아니라 범용적인 데이터도 적합한 모델을 의미한다. 즉, 잘 일반화하기 위해서는 이상치나 노이즈가 들어와도 크게 흔들리지 않아야(=robust) 한다.

    ---

    * parameters
        * 모델 내부에서 확인이 가능한 변수로 데이터를 통해 산출 가능한 값. 학습할 때 모델에 의해 요구되는 값들
        * 모델의 능력을 결정하며 주로 데이터로부터 학습되고 사용자가 정하지 않음
        * 알고리즘의 최적화 과정에서 정해짐

    * Parametric Model
        * 데이터가 특정 분포를 따른다 가정
        * 결정해야하는 파라미터의 종류와 수가 결정되어 있음
        * 우선 모델의 형태를 결정하고 모델의 파라미터를 학습을 통해 발전시키는 방식
        * Training data에 대해 우리가 가지고 있는 정보를 요약하여 W에 저장
        * Test 할 때 실제 training data를 사용할 필요 없음 ( kNN )

    * Non – Parametric Model
        * 데이터가 특정 분포를 따른다는 가정 없음 (더 flexible)
        * 학습에 따라 튜닝할 파라미터가 명확하게 정해져 있지 않음
        * Data에 대한 사전지식이 전혀 없을 때 유용
        * 더 큰 데이터를 필요로 하고 학습에 시간이 오래 걸림

    * ❗ hyperparameters
        * 모델에서 외적인 요소로, 데이터 분석을 통해 얻어지는 값이 아니라 사용자가 직접 정하는 값
        * 모델의 parameter 값을 측정하기 위해 알고리즘 구현 과정에서 사용
        * 경험에 의해 정해지는 경우가 많아 여러 번 수행해보며 최적의 값을 찾음

        ---

        * what is the best value of k to use?
        * what is the best distance to use?
        * choice about the algorithm that we set rather than learn
            * problem-dependent
            * must try them all out and see what works best.

        1. validation data
            ![image](https://user-images.githubusercontent.com/55133871/112740071-ba744780-8fb4-11eb-8d8d-8006a58cef6e.png)
            * validation 데이터와 test 데이터 사이의 엄격한 구분을 두는 것은 매우 중요
            * Training set: 모델을 학습시키는 data
            * Validation set: 최적화를 위해 Hyperparameter Tuning에 사용
            * Test set: 모델의 최종 성능 평가를 위해 사용
        
        2. cross-validation
            * 모든 데이터를 활용함. 트겅 데이터에 overfitting 되는 것을 방지. 시간이 오래 걸리기 때문에 데이터가 충분히 클 때는 사용X
            ![image](https://user-images.githubusercontent.com/55133871/112740208-083d7f80-8fb6-11eb-8f77-c9320e2ff8ac.png)
            * 강의 자료에 있는 표의 경우, seens that k ~=7 works best for this data
            * K-fold Validation
                * K개의 fold로 데이터를 분류한 후 각각 한 fold의 데이터를 validation set으로 활용해 총 K번 검증하고 평균을 냄
                * 모델의 성능을 전체적으로 볼 수 있음

    ---

    * k-Nearest Neighbor on images never used.
        * very slow at test time : test 시간이 오래 소요된다는 것은 큰 결점
        * distance metrics on pixels are not informative
            * 사람의 눈으로는 다른 이미지라는 것을 인식하는데 컴퓨터는 잘 인식하지 못함..
            * 그래서 정확한 결과를 얻기 위해 데이터 셋이 많을 수록 좋지만 물론 우리가 가지고 있는 데이터의 수는 제한적이기 때문에, 학습셋에 가짜 데이터(fake data)를 포함할 수 있다.
            * Augmentation
                * Training set과 현실의 test set 사이의 괴리감 존재
                * 임의의 잡음이나 translation을 training set에 가해서 괴리감을 줄이고 성능 향상
                * 데이터 크기가 작은 경우 데이터셋을 늘리기 위해 사용
                * flip, noise, contrast, combination
        * curse of dimensionality
            ![image](https://user-images.githubusercontent.com/55133871/112740585-0aeda400-8fb9-11eb-8281-c8544598c047.png)
            * kNN이 잘 작동하려면 data set이 공간 전체를 덮어야 함. dimension이 높아질수록 데이터 많이 필요 = 비표율적
            * the number of training examples that we need to densely cover the space grows exponentially with the dimension.

    ---

    * summary
        ![image](https://user-images.githubusercontent.com/55133871/112740420-df1dee80-8fb7-11eb-9c2e-c22f0c7ed18d.png)


### 3. 🖥 linear classifier : SUPER IMPORTANT
> 딥러닝은.. 레고와도 같다. 차곡차곡.. 여러개의 linear classifier...

![image](https://user-images.githubusercontent.com/55133871/112740874-599c3d80-8fbb-11eb-88c8-0ca1e35f3924.png)
* W
    * Training data에 대한 정보가 요약된 벡터
    * k-nearest에서는 parameter가 없고 전체 학습된 데이터를 테스트에 사용했었다.
    * 하지만~ we're going to summarize our knowledge of the training data and stick all that knowledge into these parameters, W.
    * 그렇게 되면 실제 training data들은 필요 없고 테스트할 때 W만 사용하면 된다는 말씀!
* F
    * 조건에 맞게 설정하는 함수
* b
    * training data와 상호작용하지 않는 상수
    * 예를 들어 고양이보다 강아지 사진이 훨씬 더 많을 경우 고양이의 편향(bias) 요소가 다른 것보다 더 높다.

* 이러한 모습임
    ![image](https://user-images.githubusercontent.com/55133871/112740996-97e62c80-8fbc-11eb-8242-c3b79eaa24d9.png)
    * 이미지는 3차원의 shape(RGB니까)
    * 기존의 저장되어 있던 W와 곱해서 > 이미지의 pixel values들을 하나의 긴 column vector로 만들어 1차원의 값으로 만듦
    * 그 후 필요에 따라 임의로 B를 더해주면
    * 최종적인 score가 나온다! (각 class에 대한 score이 담긴 벡터)

    * to go back to this idea of images as points and high dimensional space.
        ![image](https://user-images.githubusercontent.com/55133871/112741052-1347de00-8fbd-11eb-8f71-4d22544de4be.png)

    * 보충 필요
        ![image](https://user-images.githubusercontent.com/55133871/112741074-31add980-8fbd-11eb-92aa-1278a05a63ba.png)
        ![image](https://user-images.githubusercontent.com/55133871/112741112-746fb180-8fbd-11eb-9e07-0d0543691a37.png)