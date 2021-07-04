## ✔ CS231n 10주차 예습과제

1. RNN
    * 기본적인 one-to-one vanilla Neural Network가 연속적으로 연결된 구조를 바탕으로 합니다.
        * one-to-many : image 값을 넣었을 때 이걸 설명하는 단어들을 만들어내는 image captioning에서 많이 쓰임
        * many-to-one : 연속적인 단어들을 입력했을 때 감정을 추출해내는 sentiment classification에서 많이 쓰임
        * many-to-many : 번역 모델에서 많이 쓰임
        * many-to-many : video classification on frame level에서 많이 쓰임
    * RNN은 CNN과 달리 sequential processing을 바탕으로 만들어졌기에 문장, 비다오와 같이 sequential 한 data를 주로 잘 처리할 수 있다.
    * image와 같이 non-sequence data도 잘 처리할 수 있는데 예를 들어 숫자 image가 있다고 할 때 image를 부분적으로 보면서 처리하는 것. 이런 방식을 사용하면 non-sequence data라 하더라도 sequential processing이 가능함.

    * RNN의 기본 구조
        * hidden state를 반복적으로 거침(hidden state에 저장된 값을 다음 step에서 사용)
        * 이때 사용되는 가중치는 언제나 동일
        * 이런 구조를 갖게 된 이유
            * sequential process를 구현하기 위함. 문장을 번역한다고 했을 때 앞의 맥락을 알아야 그 뒤에 이어질 말을 예측할 수 있는 것처럼 sequential한 구조에서는 이런 식으로 이전의 data를 기억해야 함.
        * 계산 방식에 대해서는 세미나 수강 후 더 자세하게 정리해보기.
        * RNN을 이용해서 학습을 진행한다고 했을 때 시퀀스가 매우 긴 경우에는 굉장히 불리하다. 만약 책 한권을 입력값으로 넣는다면 backpropagation에 걸리는 시간이 매우매우 길 것이다. (소설책을 끝까지 다 보고 나서야 gradient를 1회 update하기 때문) 따라서 이러한 경우엔 전체 문장을 잘게 쪼게서 각 단위별로 loss를 계산하는 방식을 사용한다.
        * 셰익스피어와 수학책 예시 신기하다..
        * 보통 더 좋은 성능을 내기 위해서 RNN을 여러층 쌓아서 사용하지만 모델이 커지면 거리가 멀리 떨어진 정보끼리 gradient가 전달이 잘 안 될 수도 있다.

2. LSTM(Long Short Term Memory)
    * RNN의 장기 의존성 문제를 해결하기 위해 탄생
    * https://brunch.co.kr/@chris-song/9
    * https://ratsgo.github.io/natural%20language%20processing/2017/03/09/rnnlstm/
    * 나중에 읽어보기