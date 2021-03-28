## ✔ CS231n 2주차 예습과제

(5:00~)

### 1. 😺 Image Classification pipeline
* 이미지 분류는 컴퓨터 비전의 가장 메인 task 중 하나
    * input image -> 라벨링 되어 있는 이미지군들을 사용해 들어온 이미지를 인지(카테고리에 할당)
    * 이미지가 800x600 픽셀이면 각 픽셀은 RGB를 표현하기 위한 3자리의 수로 이루어져 있음 (= 매우 많은 양의 수들!)
    * 이 수들로 라벨링된 데이터에 따라 이 이미지가 어떤 군에 속하는지 분류하는 것은 사실 매우 어려운 일이다. (물체가 바뀌는 것은 물론, 대상이 같아도 카메라가 움직이거나 빛의 위치가 달라지면 픽셀값이 달라진다.)

* Challenges
    * Background Clutter
        * 물체가 배경과 매우 흡사한 경우
    * Intraclass variation
        * 같은 고양이라도 모습은 모두 다르니

* How?
    1. Attempts have been made
        * find edges(compute the edges of image) -> try to categorize all the different corners and boundaries...
        * 다루기 힘들고, 이 방대한 작업을 카테고리마다 전부 실행해야 함.
    2. Data-Driven Approach
        1. 인터넷에서 매우 많은 양의 데이터셋(이미지)를 모은다
        2. 데이터들을 학습을 통해 구분하고 정리해서 모델(이러한 이미지 카테고리들을 어떻게 인식/분류하는지에 대해 학습한 지식을 요약한 것)을 만든다
        3. 이 모델을 사용해 새로운 이미지를 인식
        ---
        * train
            * input images and labels output a model
        * predict
            * input the model and make predictions for images


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

* K-Nearest Neighbors : Distance Metric
    ![image](https://user-images.githubusercontent.com/55133871/112739818-72542580-8fb2-11eb-93df-6ee20bbb1676.png)
    * L1(Manhattan) distance
    * L2(Euclidean) distance

    * 이미지를 분류하는 작업 이외에도 문장 유사도 등 다양한 것을 분류하는데 이 알고리즘을 확장해 사용할 수 있다.

    ---

    * ❗ hyperparameters
        * what is the best value of k to use?
        * what is the best distance to use?
        * choice about the algorithm that we set rather than learn
            * problem-dependent
            * must try them all out and see what works best.

        1. validation data
            ![image](https://user-images.githubusercontent.com/55133871/112740071-ba744780-8fb4-11eb-8d8d-8006a58cef6e.png)
            * validation 데이터와 test 데이터 사이의 엄격한 구분을 두는 것은 매우 중요
        
        2. cross-validation
            ![image](https://user-images.githubusercontent.com/55133871/112740208-083d7f80-8fb6-11eb-8f77-c9320e2ff8ac.png)
            * 작은 데이터셋에만 쓰이고 딥러닝에는 잘 안 쓰임
            * 세미나 듣고 질문 부분 다시 돌려보기
            * 강의 자료에 있는 표의 경우, seens that k ~=7 works best for this data

    ---

    * k-Nearest Neighbor on images never used.
        * very slow at test time
        * distance metrics on pixels are not informative
            * 이미지의 차이를 비교하는데 좋은 방법이 아님.
            * 같은 사람을 두고 여러가지 변형을 주어도 차이점을 잘 잡아내지 못한다(강의자료)
        * curse of dimensionality
            ![image](https://user-images.githubusercontent.com/55133871/112740585-0aeda400-8fb9-11eb-8281-c8544598c047.png)
            * 이 부분 조금 더 찾아보기
            * the number of training examples that we need to densely cover the space grows exponentially with the dimension.

    ---

    * summary
        ![image](https://user-images.githubusercontent.com/55133871/112740420-df1dee80-8fb7-11eb-9c2e-c22f0c7ed18d.png)


### 3. 🖥 linear classifier : SUPER IMPORTANT
> 딥러닝은.. 레고와도 같다. 차곡차곡.. 여러개의 linear classifier...

![image](https://user-images.githubusercontent.com/55133871/112740874-599c3d80-8fbb-11eb-88c8-0ca1e35f3924.png)
* W
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

    * to go back to this idea of images as points and high dimensional space.
        ![image](https://user-images.githubusercontent.com/55133871/112741052-1347de00-8fbd-11eb-8f71-4d22544de4be.png)

    * 보충 필요
        ![image](https://user-images.githubusercontent.com/55133871/112741074-31add980-8fbd-11eb-92aa-1278a05a63ba.png)
        ![image](https://user-images.githubusercontent.com/55133871/112741112-746fb180-8fbd-11eb-9e07-0d0543691a37.png)