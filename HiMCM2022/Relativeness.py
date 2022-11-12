import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set(color_codes=True)
# 支持显示中文
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
data = pd.read_excel('HIMCM data.xlsx')

dsc = data.describe()
print(dsc)

pd.plotting.scatter_matrix(data, figsize=(20,10), alpha=0.75)
plt.show()

cor = data.corr()  # 默认method='pearson'
print(cor)


# import seaborn as sns
# 支持中文显示

fig, ax = plt.subplots(figsize = (10,10))

# cor：相关系数矩阵
# cmap：颜色
# xticklabels：显示x轴标签
# yticklabels：显示y轴标签
# annot：方块中显示数据
# square：方块为正方形

sns.heatmap(cor, cmap='YlGnBu', xticklabels=True, yticklabels=True,
            annot=True, square=True)
plt.show()


from scipy import stats

np.set_printoptions(suppress=True)  # 不使用用科学计数法
pd.set_option('display.float_format', lambda x: '%.4f' % x)  # 保留小数点后4位有效数字

# 0.975分位数
tp = stats.t.isf(1 - 0.975, 62)

x = np.linspace(-5, 5, 100)
y = stats.t.pdf(x, 62)
plt.plot(x, y)
plt.vlines(-tp, 0, stats.t.pdf(-tp, 62), colors='orange')
plt.vlines(tp, 0, stats.t.pdf(tp, 62), colors='orange')
plt.fill_between(x, 0, y, where=abs(x) > tp, interpolate=True, color='r')


# 自定义求解p值矩阵的函数
def my_pvalue_pearson(x):
    col = x.shape[1]
    col_name = x.columns.values
    p_val = []
    for i in range(col):
        for j in range(col):
            p_val.append(stats.pearsonr(x[col_name[i]], x[col_name[j]])[1])
    p_val = pd.DataFrame(np.array(p_val).reshape(col, col), columns=col_name, index=col_name)
    p_val.to_csv('p_val_pearson.csv')  # 此处实则为多此一举，目的是借助带有excel格式的数据使得输出更美观
    p_val = pd.read_csv('p_val_pearson.csv', index_col=0)
    return p_val

a = my_pvalue_pearson(data)
print(a)

data.corr(method='spearman')

# 自定义求解p值矩阵的函数
def my_pvalue_spearman(x):
    col = x.shape[1]
    col_name = x.columns.values
    p_val = []
    for i in range(col):
        for j in range(col):
            p_val.append(stats.spearmanr(x[col_name[i]], x[col_name[j]])[1])
    p_val = pd.DataFrame(np.array(p_val).reshape(col, col), columns=col_name, index=col_name)
    p_val.to_csv('p_val_spearman.csv')  # 此处实则为多此一举，目的是借助带有excel格式的数据使得输出更美观
    p_val = pd.read_csv('p_val_spearman.csv', index_col=0)
    return p_val

b = my_pvalue_spearman(data)
print(b)

SAMPLE_NUM = 64
print(data, 64)

# 先预设一个结果，假定拟合的结果为 y=-6x+10
X = np.linspace(-10, 10, 64)
a = -6
b = 10
Y = list(map(lambda x: a * x + b, X))
print("标准答案为：y={}*x+{}".format(a, b))

# 增加噪声，制造数据
Y_noise = list(map(lambda y: y + np.random.randn() * 10, Y))
plt.scatter(X, Y_noise)
plt.title("data to be fitted")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

A = np.stack((X, np.ones(SAMPLE_NUM)), axis=1)  # shape=(SAMPLE_NUM,2)
b = np.array(Y_noise).reshape((SAMPLE_NUM, 1))

print("方法列表如下:"
      "1.最小二乘法 least square method "
      "2.常规方程法 Normal Equation "
      "3.线性回归法 Linear regression")
method = int(input("请选择您的拟合方法: "))

Y_predict = list()
if method == 1:
    theta, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    # theta=np.polyfit(X,Y_noise,deg=1) 也可以换此函数来实现拟合X和Y_noise,注意deg为x的最高次幂，线性模型y=ax+b中，x最高次幂为1.
    # theta=np.linalg.solve(A,b) 不推荐使用
    theta = theta.flatten()
    a_ = theta[0]
    b_ = theta[1]
    print("拟合结果为: y={:.4f}*x+{:.4f}".format(a_, b_))
    Y_predict = list(map(lambda x: a_ * x + b_, X))

elif method == 2:
    AT = A.T
    A1 = np.matmul(AT, A)
    A2 = np.linalg.inv(A1)
    A3 = np.matmul(A2, AT)
    A4 = np.matmul(A3, b)
    A4 = A4.flatten()
    a_ = A4[0]
    b_ = A4[1]
    print("拟合结果为: y={:.4f}*x+{:.4f}".format(a_, b_))
    Y_predict = list(map(lambda x: a_ * x + b_, X))

elif method == 3:
    # 利用线性回归模型拟合数据，构建模型
    model = LinearRegression()
    X_normalized = np.stack((X, np.ones(SAMPLE_NUM)), axis=1)  # shape=(50,2)
    Y_noise_normalized = np.array(Y_noise).reshape((SAMPLE_NUM, 1))  #
    model.fit(X_normalized, Y_noise_normalized)
    # 利用已经拟合到的模型进行预测
    Y_predict = model.predict(X_normalized)
    # 求出线性模型y=ax+b中的a和b，确认是否和我们的设定是否一致
    a_ = model.coef_.flatten()[0]
    b_ = model.intercept_[0]
    print("拟合结果为: y={:.4f}*x+{:.4f}".format(a_, b_))

else:
    print("请重新选择")

plt.scatter(X, Y_noise)
plt.plot(X, Y_predict, c='green')
plt.title("method {}: y={:.4f}*x+{:.4f}".format(method, a_, b_))
plt.show()





#%%
