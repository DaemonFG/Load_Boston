"""
欠拟合：一个假设在训练数据上不能获得更好的拟合，但是在训练数据外的数据集上也不能很好地拟合数据，
此时认为这个假设出现了欠拟合的现象。(模型过于简单)
原因：
学习到数据的特征过少
解决办法：
增加数据的特征数量

过拟合：一个假设在训练数据上能够获得比其他假设更好的拟合，但是在训练数据外的数据集上却不能很好地拟合数据，
此时认为这个假设出现了过拟合的现象。(模型过于复杂)
原因：
原始特征过多，存在一些嘈杂特征，模型过于复杂是因为模型尝试去兼顾各个测试数据点。

解决办法：
进行特征选择，消除关联性大的特征(很难做)
交叉验证(让所有数据都有过训练)
正则化

特征选择：
过滤式：低方差特征
嵌入式：正则化，决策树，神经网络

l2正则化：
作用：可以使得W的每个元素都很小，都接近于0
优点：越小的参数说明模型越简单，越简单的模型则越不容易产生过拟合现象

Ridge岭回归：
带有正则化的线性回归，解决过拟合

sklearn.linear_model.Ridge(alpha=1.0)
具有l2正则化的线性最小二乘法
alpha:正则化力度λ
coef_:回归系数

"""

from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


def mylinear():
    """
    线性回归直接预测房价
    :return: None
    """
    # 获取数据
    lb = load_boston()

    # 分割训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)

    # 进行标准化处理
    # 特征值和目标值都需要进行标准化处理，因为特征值和目标值特证数不同，所以需要实例化两个标准化API
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)

    # 目标值
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train.reshape(-1, 1))  # 0.19版本以后要求是二维，不知道样本数，所以-1，一个特征所以1
    y_test = std_y.transform(y_test.reshape(-1, 1))

    # estimstor预测
    # 正规方程求解方式预测结果
    rg = Ridge()
    rg.fit(x_train, y_train)
    print("正规方程回归系数：", rg.coef_)

    # 预测的房价
    y_predict = std_y.inverse_transform(rg.predict(x_test))
    print("正规方程测试集里每个房子的预测价格：", y_predict)
    print("正规方程的均方误差：", mean_squared_error(std_y.inverse_transform(y_test), y_predict))

if __name__ == '__main__':
    mylinear()
