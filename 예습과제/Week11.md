## ✔ CS231n 11주차 예습과제

#### Image Segmentation

- Image segmentation

  - image 단위로 labeling을 하는 것이 아니라, 모든 pixel에 대해 independently labeling하여 pixel 단위로 classification하는 것을 의미
  - 붙은 2개의 물체를 한 개로 인식함. 이 문제를 해결한 것이 마지막 문제인 instance segmentation.
  - conv를 이용해 high resolution image를 계속 processing 해서 segmentation하는 방법이 고안되었으나, 계속해서 high resolution image를 sequence로 처리하는 것은 cost가 높다.

- Downsampling하고 Upsampling하여 high resolution image로 복구하는 것

  - FC를 통해 transitioning을 하는 것과는 달리 spatial resolution을 계속 증가시키는 방식. prediction이 ground truth에 근접하도록 학습하는 방법

- Transpose Convolution

  - Convolution연산을 할 때 stride를 넣어주면 down smapling되는 현상이 발생
  - 반대로 1개의 pixel에다 vector(kernel)를 곱해 여러 개의 값을 뽑아내고 stride를 주고 움직여서 image size를 키우는 방식. overlap되는 부분에서는 평균을 취하지 않고 더해주기만 함.

- Object Detection

  - Localization과 달리 여러 개의 object에 대해 annotation을 하는 것. 매번 다른 crop을 만들고, sliding하고 하는 방식의 brute force는 굉장히 cost가 큼. 그래서 object가 있는 곳을 먼저 proposal을 받는 알고리즘을 사용.
  - Region Proposals(fixed way)

- R-CNN

  - fixed way regin proposal network을 사용해 candidate를 뽑고 CONV로 처리
  - region proposal에 굉장한 시간을 쏟게 되며, candidate가 많으면 그만큼 학습량도 증가.

- Fast R-CNN

  - CONV로 high resolution feature map을 얻은 뒤 fixed way selective search로 crop 구함

- Faster R-CNN

  - CNN으로 feature map을 딸때 proposal을 같이 학습. region proposal또한 learnable part가 되어 빠르게 forwarding 가능
  - faster R-CNN style의 region-based method가 single shot보다 더 높은 정확도를 가짐. 각 region에 대해 독립적으로 processing하지 않아도 되기 때문에 훨씬 빠름.

(12강 발표로 11강 내용만 적음.)​
