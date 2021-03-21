## ✔ CS231n 1주차 예습과제

### 1. 🎓 CS231n
* learn about neural network(deep learning) class on image classification

### 2. 🖥 HISTORY
* "what was the visual processing mechanism like in mammals?"
    * visual processing starts with small cells.

--- 

* 60s - David Mars
    * input image > Edge image > 2 1/2-D sketch > 3-D model

* 70s - every object is composed of simple geometric primitives
    * "how can we start recognizing or representing real world objects?"
    * generalized cylinder
    * pictorial structure

* 80s - David Lowe
    * tries to recognize razors by constructing lines, edges, straight lines and their combination.

--- 

* Image Segmentation
    * "if object recognition is too hard, maybe we should first do object segmentation! group the images into meaningful areas."
    
* Late 90s - 2010
    * SIFT feature : match the entire object
        * 카메라의 초점, 배경, 빛 모든 조건이 달라도 물체의 변하지 않는 요소(특징)들을 비교해서 판단 > 물체 전체의 패턴을 찾는 것보다 쉽다
    * 장면에 대한 힌트를 주는 특징을 찾아라!

* Early 2000
    * benchmark data set : PASCAL Visual Object Challenge
    * performance on detecting objects in benchmark data set has steadily increased.
    * 인터넷의 발전에 따라 quality of pictures가 좋아진 것도 한몫함

* 모든 물체를 구분할 수 있는가?
    * ImageNet : 많은 데이터 셋 > 의미 있는 에러율 감소 > 이미지 분류 외 다른 분야에도 적용 가능

### 3. ✨ OVERVIEW
* CNN(Convolution Neural Networks)
    * 2012년이 엄청난 해였구나
    * computation : 컴퓨터의 능력이 향상됨에 따라 딥러닝의 영역이 확장될 수 있었다
    * data : 인터넷 발전 전에는 퀄리티 높고 방대한 데이터를 갖추기 힘들었다
    * 암튼 이러한 알고리즘은 새로운 것이 아니라 오래 전부터 있었다는 것임
    * 아주 멋진 사진을 예시로 드시네.. TA 진짜 많다 학생 수가 많아서 그런가 오..