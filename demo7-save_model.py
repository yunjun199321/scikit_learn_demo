import pickle  # pickle模块

from sklearn import datasets
from sklearn import svm
from sklearn.externals import joblib  # joblib模块

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)

# 保存Model(注:save文件夹要预先建立，否则会报错)
with open('save/clf.pickle', 'wb') as f:
    pickle.dump(clf, f)

# 读取Model
with open('save/clf.pickle', 'rb') as f:
    clf2 = pickle.load(f)
    # 测试读取后的Model
    print(clf2.predict(X[0:1]))

# ------------------------------------------------------------

# 保存Model(注:save文件夹要预先建立，否则会报错)
joblib.dump(clf, 'save/clf.pkl')

# 读取Model
clf3 = joblib.load('save/clf.pkl')

# 测试读取后的Model
print(clf3.predict(X[0:1]))
