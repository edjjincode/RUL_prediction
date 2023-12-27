# Psyence(인공지능 기초 솔루션 기업)과 협업하여 Nasa Turbofan 데이터를 활용하여 RUL prediction README 파일

## Description

### 📝 프로젝트 설명:

본 프로젝트는 NASA에서 제공하는 NASA Turbofan Engine 데이터를 사용하여 잔여수명에 사용되는 대표적인 3가지 방법들을 활용하여 잔여수명을 예측하고 예측한 결과를 바탕으로더 좋은 방법을 제시하고자 한다. 더 나아가 산학 협력 관계를 맺고 같이 프로젝트를 진행한 Pysence 기업에 다변량 시계열 데이터에서 잔여수명 예측을 할 때 사용될 수 있는 머신러닝 모델을 제시하여 인사이트를 제공하고자 한다.

### 📈 잔여 수명 예측에 사용되는 3가지 방법론:

잔여 수명 예측은 통상적으로 3가지 방법론을 사용한다. 첫 번째 방법은 유사성 모델을 활용하는 것이다. 유사성 모델은 비슷한 동작을 보이는, 유사하거나 다른 구성 요소의 Run-To-Failure 데이터를 활용해 RUL을 추정하는 방식이다. 두 번째 방법은 생존 모델을 활용하는 것이다. 생존 모델은 수명 데이터가 주어졌을 때 사용된다. 마지막 세 번째 방법은 건전 지표를 활용하는 것이다. 이때는 도메인에 따라 규정된 임계 값이 주어졌을 때 사용될 수 있다. 

### 📊 잔여 수명 예측 결과:

Nasa Turbofan 데이터 셋에 위 3가지 방법들을 모두 적용시킨 결과, 유사성 모델이 가장 성능이 좋게 나타나고 생존 모델, 건전 지표 모델 순으로 성능이 좋은 것을 알 수 있다.


## 협업 프로젝트 진행 후 코드 패키지화 진행

처음 프로젝트를 진행할 때는 주피터 노트북에서 함수형 프로그래밍을 활용하여 코드를 진행하였다. 프로젝트를 진행한 후, 코드 가독성을 높이기 위해 코드를 패키지화 시킨 후 코드를 진행할 수 있도록 하였다. 폴더 중에 Before 폴더 같은 경우, 주피터 노트북 형태의 코드들이 있고, After 폴더 같은 경우, 패키지화 된 코드, Data 폴더 같은 경우 프로젝트에 사용된 데이터를 넣었다.

---

## 🖼️ Environment(requirements.txt)

requirements.txt 파일을 참고하면 된다.

---

## 📂 Files

```
|------Before
|  |-- Similarity_model_dwt.ipynb
|  |-- Similarity_model.ipynb
|  |-- Exponential_Degradation.ipynb
|  |-- Exponential_Degradation_dwt.ipynb
|  |-- Survival_Model.ipynb
|  |-- Survival_Model_dwt.ipynb
|
|------After
|  |--Exponential_Degradation
|  |  |-- basic
|  |  |---- Exponential_Degradation
|  |  |------- __init__.py
|  |  |-------_pycache__
|  |  |------- data_preparation
|  |  |          |--__init__.py
|  |  |          |--prepare_data.py
|  |  |          |--process_data.py
|  |  |------- model
|  |  |          |--__init__.py
|  |  |          |--exp_degradation.py
|  |  |          |--pca.py
|  |  |------- utils
|  |  |          |--__init__.py
|  |  |          |--evaluation.py
|  |  |---- main.py
|  |  |-- dwt
|  |  |---- Exponential_Degradation
|  |  |------- __init__.py
|  |  |-------_pycache__
|  |  |------- data_preparation_dwt
|  |  |          |--__init__.py
|  |  |          |--prepare_data.py
|  |  |          |--process_data.py
|  |  |------- model_dwt
|  |  |          |--__init__.py
|  |  |          |--dwt.py
|  |  |          |--exp_degradation.py
|  |  |          |--pca.py
|  |  |------- utils_dwt
|  |  |          |--__init__.py
|  |  |          |--evaluation.py
|  |  |---- main.py
|  |--SimilarityModel
|  |  |-- basic
|  |  |---- SimilarityModel
|  |  |------- __init__.py
|  |  |-------_pycache__
|  |  |------- data_preparation
|  |  |          |--__init__.py
|  |  |          |--prepare_data.py
|  |  |          |--process_data.py
|  |  |------- health_indicator
|  |  |          |--__init__.py
|  |  |          |--health_indicator.py
|  |  |          |--similarity_score.py
|  |  |------- utils_dwt
|  |  |          |--__init__.py
|  |  |          |--evaluation_dwt.py
|  |  |---- main.py
|  |  |-- dwt
|  |  |---- SimilarityModel
|  |  |------- __init__.py
|  |  |-------_pycache__
|  |  |------- data_preparation_dwt
|  |  |          |--__init__.py
|  |  |          |--prepare_data.py
|  |  |          |--process_data.py
|  |  |------- health_indicator_dwt
|  |  |          |--__init__.py
|  |  |          |--dwt_health_indicator.py
|  |  |          |--dwt.py
|  |  |          |--similarity_score_dwt.py
|  |  |------- utils_dwt
|  |  |          |--__init__.py
|  |  |          |--evaluation_dwt.py
|  |  |---- main.py
|  |--SurvivalModel
|  |  |-- basic
|  |  |---- Survival_Model
|  |  |------- __init__.py
|  |  |-------_pycache__
|  |  |------- data_preparation
|  |  |          |--__init__.py
|  |  |          |--prepare_data.py
|  |  |          |--process_data.py
|  |  |------- model
|  |  |          |--__init__.py
|  |  |          |--CoxPHModel.py
|  |  |------- utils
|  |  |          |--__init__.py
|  |  |          |--evaluation.py
|  |  |---- main.py
|  |  |-- dwt
|
|-----Data
|  |  |--RUL_FD001.txt
|  |  |--test_FD001.txt
|  |  |--train_FD001.txt
|
|  |-- README.md
|  |-- requirements.txt

```

