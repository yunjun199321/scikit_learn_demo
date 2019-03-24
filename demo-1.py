from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data  # 4 attributes
iris_Y = iris.target  # 3 types of iris

# print(iris_X[:2, :])
# print(iris_Y)

# 把数据集分成training set 和 target set. x_test + y_test = 30%
# 当split了之后，数据会被打乱。这样更有利于机器学习
x_train, x_test, y_train, y_test = train_test_split(iris_X, iris_Y, test_size=0.3)

# print(y_train)
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)  # 这里最重要，把数据都放进去训练。 knn 经过这步就是训练完的模型了。

print(knn.predict(x_test))  # 用predict测试数据
print(y_test)
