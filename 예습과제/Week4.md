## ✔ CS231n 4주차 예습과제

### Title
Backpropagation(역전파법) and Neural Networks(인공신경망)

<br/>

### 1. Backpropagation

- scores function으로 각 클래스의 점수를 구한 후, 그 점수를 사용해 SVM Loss를 구한 후, regularization을 더해 최종 Loss를 구한다. 이 Loss를 가장 적게 만든 후
- 그 이후 조금씩 loss가 내려가는 방향으로 분류기를 수정, 기울기를 줄인다!(optimization)
- 그런데... 함수가 복잡해질 수록 그래프가 엄청나게 복잡해진다!
- 그래서 등장한 Backpropagation(역전파법)

![1](https://user-images.githubusercontent.com/55133871/116809794-ddc58000-ab7a-11eb-80a4-aa1265f6f5ff.PNG)

* 입력값을 받아서 loss 값을 구하기까지 계산해 가는 과정을 forward pass
* forward pass 종료 이후 역으로 미분하며 기울기 값들을 구해가는 과정을 backward pass라고 한다.

![2](https://user-images.githubusercontent.com/55133871/116809843-4280da80-ab7b-11eb-9aba-e0fcdc7778a9.PNG)

<br/>

![3](https://user-images.githubusercontent.com/55133871/116809880-76f49680-ab7b-11eb-9d8b-36dcfccc053d.PNG)
sigmoid gate // 자주 사용되는 임의의 부분만 떼와서 해당 부분만 쉽게 계산할 수 있도록 함.

<br/>

backpropagation에서 자주 보이는 패턴들
![4](https://user-images.githubusercontent.com/55133871/116809985-2467aa00-ab7c-11eb-809a-3d037abcca54.PNG)
* add gate : 기존에 가지고 있던 gradient를 각각의 노드에 분배
* mul gate : 현재의 gradient를 각각 숫자에 곱해서 위치를 바꿔줌
* max gate : 더 큰 쪽에 gradient를 그대로 두고, 반대쪽은 제거됨

<br/>

![5](https://user-images.githubusercontent.com/55133871/116810208-7230e200-ab7d-11eb-8851-6f2e7227a8fd.PNG)

* 인공 신경망은 굉장히 거대하다. 따라서 미분 값들을 모두 손으로 계산하는 것은 불가능하기 때문에 역전파법을 사용하는 것
* 그 과정에서 chain rule을 사용해 모든 입력값 및 변수들에 대한 기울기를 계산
* forward : gate들을 정방향으로 거쳐가며 계산한 값을 메모리에 저장하는 것
* backward : 그 값들을 기반으로 역방향으로 계산하며 chain rule을 적용시켜 input 값이 loss 값에 어떤 영향을 미치는지 기울기 값을 계산하는 것

<br/>

Q) 거꾸로 계산하는 이유?
A) 정방향으로 계산하면서 하나씩 미분 값을 계산하는 것은 연산 낭비가 심함

<br/>

### 2. Neural Networks

![6](https://user-images.githubusercontent.com/55133871/116810455-d902cb00-ab7e-11eb-8384-e39867d6b261.PNG)

* 인공 신경망에 대한 설명을 다시 한 번 들어봐야 될 것 같다..흠

<br/>

![7](https://user-images.githubusercontent.com/55133871/116810519-25e6a180-ab7f-11eb-80ca-b99b57ae35a6.PNG)