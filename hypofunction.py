"""
이번 과제에서는 가설 함수를 사용해서 주어진 데이터를 예측하는 코드를 구현해보겠습니다. 
prediction이라는 함수로 구현할 건데요. 이 함수에 대해서 설명드릴게요.

prediction 함수
prediction 함수는 주어진 가설 함수로 얻은 결과를 리턴하는 함수입니다. 파라미터로는 \theta_0θ 
0 를 나타내는 숫자형 변수 theta_0, \theta_1θ 
1
​
 를 나타내는 숫자형 변수 theta_1, 그리고 모든 입력 변수 벡터 xx들을 나타내는 numpy 배열 x를 받죠.

가설 함수를 h_\theta(x) = \theta_0 + \theta_1xh 
θ
​
 (x)=θ 
0
​
 +θ 
1
​
 x 이렇게 정의했는데요.

prediction 함수는 x의 각 요소의 예측값에 대한 numpy 배열을 리턴합니다.

numpy 배열과 연산들을 이용해서 prediction 함수를 작성해보세요.

numpy 배열과 숫자형 덧셈
numpy 배열과 일반 숫자형을 더하면 numpy 배열의 모든 요소에 해당 숫자형이 더해집니다. 이걸 사용해서 과제를 풀어보세요!

np_array = np.array([1, 2, 3, 4, 5])
    
5 + np_array  # [6, 7, 8, 9, 10]
출력 예시
array([ -1.2,  -0.2,   1. ,   1.2,   2.2,   3.6,   3.7,   4.8,   5.8,
         6.4,   7.4,   8.5,  10.4,  10.8])
"""

import numpy as np

def prediction(theta_0, theta_1, x):
    """주어진 학습 데이터 벡터 x에 대해서 예측 값을 리턴하는 함수"""
    # 코드를 쓰세요
    # h(x) = @0 + @1v[x]
    return  theta_0 + theta_1 * x
    
def prediction_difference(theta_0, theta_1, x, y):
    """모든 예측 값들과 목표 변수들의 오차를 벡터로 리턴해주는 함수"""
    # 코드를 쓰세요

    return prediction(theta_0, theta_1, x) - y
# 테스트 코드

# 입력 변수(집 크기) 초기화 (모든 집 평수 데이터를 1/10 크기로 줄임)
house_size = np.array([0.9, 1.4, 2, 2.1, 2.6, 3.3, 3.35, 3.9, 4.4, 4.7, 5.2, 5.75, 6.7, 6.9])

# 목표 변수(집 가격) 초기화 (모든 집 값 데이터를 1/10 크기로 줄임)
house_price = np.array([0.3, 0.75, 0.45, 1.1, 1.45, 0.9, 1.8, 0.9, 1.5, 2.2, 1.75, 2.3, 2.49, 2.6])

theta_0 = -3
theta_1 = 2

print(prediction_difference(-3, 2, house_size, house_price))
