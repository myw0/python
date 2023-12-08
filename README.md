import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("/content/drive/MyDrive/A회사_보습제_매출데이터_v1.csv")
# 데이터프레임의 기본 정보 확인
print(data.info())

# 결측치 처리
data = data.dropna()

# 이상치 처리
data = data[data['column_name'] < threshold]

# 히스토그램
sns.histplot(data['column_name'], kde=True)
plt.show()

# 기초 통계량
print(data.describe())

# 선형 회귀 모델 학습
X = data[['feature1', 'feature2']]
y = data['target']
model = LinearRegression()
model.fit(X, y)


# 데이터 시각화 (히스토그램)
plt.figure(figsize=(8, 6))
plt.bar(column_data.index, column_data)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Data Visualization')
plt.show()
# 숫자 데이터 시각화 (예: 선 그래프)
plt.figure(figsize=(8, 6))
plt.plot(numeric_column_data, marker='o')
plt.xlabel('Index')
plt.ylabel('Numeric Value')
plt.title('Numeric Data Visualization')
plt.show()

