## ✔7주차 복습과제 해설
> 다음의 흐름을 따라 과제를 해결하시면 됩니다. 각 과제별로 파일을 확인해보시면 한글 주석(설명)을 달아놓았습니다.

---

### FullyConnectedNets.ipynb

#### 1. layers.py
FullyConnectedNets.ipynb 파일의 과제를 실행하는데 필요한 사전 작업입니다. affine_forward, affine_backward, relu_forward, relu_backward, softmax_loss 등 각 과제마다 필요한 함수를 미리 구현해놓으셔야 하는데, 이것은 assignment1에서 구현한 것을 재사용하셔도 됩니다.

#### 2. fc_net.py
FullyConnectedNet을 실행하기 위한 파일입니다. 파일 내부에 주석으로 설명을 달아놓았습니다.

#### 3. optim.py
7주차에서 학습했던 다양한 optimzer들을 함수로 구현하는 작업이 필요합니다. 발표 시 코드를 모두 다뤘기 때문에 별도의 설명은 불필요하다고 판단해 파일 내 별다른 주석처리는 하지 않았습니다. sgd_momentum, rmsprop, adam 함수를 완성하시면 됩니다. 단, adam의 경우 FullyConnectedNets.ipynb 파일의 문제를 확인해보면 아시겠지만 complete 버전으로 작성하라는 조건이 있습니다. 각 함수를 완성하신 후, FullyConnectedNets.ipynb 파일에서 실행해보며 잘 작동되는지 확인해보시면 됩니다.

#### 4. FullyConnectedNets.ipynb
데이터를 로드하시고, 과제의 문제를 따라 실행하시면 됩니다. SGD+Momentum 부분부터 optim.py로 이동해 함수를 작성해놓으셔야 진행이 가능합니다. Inline Question의 경우 파일의 Answer에 설명을 적어놓았습니다.

---

### BatchNormalization.ipynb

#### 1. layers.py
BatchNormalization.ipynb 파일의 과제를 실행하는데 필요한 사전 작업입니다. batchnorm_forward, batchnorm_backward, batchnorm_backward_alt, layernorm_forward, layernorm_backward 함수를 구현해야 합니다.

#### 2. fc_net.py
batch norm 과정을 추가해야 합니다. 파일 내부에 주석으로 설명을 달아놓았습니다.

#### 3. BatchNormalization.ipynb
데이터를 로드하시고, 과제의 문제를 따라 실행하시면 됩니다. 상단에는 Batch norm에 대한 설명이 적혀있습니다. Inline Question의 경우 파일의 Answer에 설명을 적어놓았습니다.

---

### Dropout.ipynb

#### 1. layers.py
Dropout.ipynb 파일의 과제를 실행하는데 필요한 사전 작업입니다. dropout_forward, dropout_backward를 구현해야 합니다.

#### 2. fc_net.py
fc_net에서 dropout을 사용하도록 수정해야 합니다. 파일 내부에 주석으로 설명을 달아놓았습니다.

#### 3. Dropout.ipynb
데이터를 로드하시고, 과제의 문제를 따라 실행하시면 됩니다. Inline Question의 경우 파일의 Answer에 설명을 적어놓았습니다.