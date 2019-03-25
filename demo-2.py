from __future__ import print_function

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression

## 训练模型
loaded_data = datasets.load_boston()
data_x = loaded_data.data
data_y = loaded_data.target

model = LinearRegression()
model.fit(data_x, data_y)

print(model.predict(data_x[:4, :]))
print(data_y[:4])

##创建虚拟数据可视化
x, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=50)

plt.scatter(x, y)
plt.show()

## 这是这个model自带的属性，斜率和截距（与Y轴）
print(model.coef_)
print(model.intercept_)

## 取出之前定义的参数
print(model.get_params())

## 对model用R^2 的方式打分
print(model.score(data_x, data_y))
