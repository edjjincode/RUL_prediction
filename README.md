# Psyence(인공지능 기초 솔루션 기업)과 협업하여 Nasa Turbofan 데이터를 활용하여 RUL prediction README 파일

## Description

### 프로젝트 설명:

본 프로젝트는 NASA에서 제공하는 NASA Turbofan Engine 데이터를 사용하여 잔여수명에 사용되는 대표적인 3가지 방법들을 활용하여 잔여수명을 예측하고 예측한 결과를 바탕으로더 좋은 방법을 제시하고자 한다. 더 나아가 산학 협력 관계를 맺고 같이 프로젝트를 진행한 Pysence 기업에 다변량 시계열 데이터에서 잔여수명 예측을 할 때 사용될 수 있는 머신러닝 모델을 제시하여 인사이트를 제공하고자 한다.

### 잔여 수명 예측에 사용되는 3가지 방법론:

잔여 수명 예측은 통상적으로 3가지 방법론을 사용한다. 첫 번째 방법은 유사성 모델을 활용하는 것이다. 유사성 모델은 비슷한 동작을 보이는, 유사하거나 다른 구성 요소의 Run-To-Failure 데이터를 활용해 RUL을 추정하는 방식이다. 두 번째 방법은 생존 모델을 활용하는 것이다. 생존 모델은 수명 데이터가 주어졌을 때 사용된다. 마지막 세 번째 방법은 건전 지표를 활용하는 것이다. 이때는 도메인에 따라 규정된 임계 값이 주어졌을 때 사용될 수 있다. 

### 잔여 수명 예측 결과:

Nasa Turbofan 데이터 셋에 위 3가지 방법들을 모두 적용시킨 결과, 유사성 모델이 가장 성능이 좋게 나타나고 생존 모델, 건전 지표 모델 순으로 성능이 좋은 것을 알 수 있다.

---

## Environment

실행환경에 대해 적는다

## Prerequisite

작성한 코드를 실행하기 전에 설치해야 할 package나 의존성이 걸리는 문제들을 설명하면 된다

## Files

각 파일들이 어떤 것을 하는 지 알려줘야 한다.

## Usage
