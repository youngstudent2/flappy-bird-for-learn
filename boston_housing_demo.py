from keras.datasets import boston_housing #导入波士顿房价数据集
from keras.models import Sequential #顺序模型
from keras.layers import Dense

(train_x, train_y), (test_x, test_y) = boston_housing.load_data()

print(train_x.shape)
print(test_x.shape)
print(train_y.shape)
print(test_y.shape)
print(train_x[0])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
train_x = sc.fit_transform(train_x)
test_x = sc.fit_transform(test_x)
print(train_x[0])
print(test_x[0])

n_hidden_1 = 64 #隐藏层1的神经元个数
n_hidden_2 = 64 #隐藏层2的神经元个数
n_input = 13 #输入层的个数
n_classes = 1 #输出层的个数（我们只预测房价，就1个值所以输出是1）
training_epochs = 200 #训练次数，总体数据需要循环多少次
batch_size = 10  #每批次要取的数据的量，这里是提取10条数据

model = Sequential()#先建立一个顺序模型
#向顺序模型里加入第一个隐藏层，第一层一定要有一个输入数据的大小，需要有input_shape参数
#model.add(Dense(n_hidden_1, activation='relu', input_shape=(n_input,)))
model.add(Dense(n_hidden_1, activation='relu', input_dim=n_input)) #这个input_dim和input_shape一样，就是少了括号和逗号
model.add(Dense(n_hidden_2, activation='relu'))
model.add(Dense(n_classes)) #因为我们是预测房价，不是分类，所以最后一层可以不用激活函数

import keras.backend as K
def r2(y_true, y_pred):
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f

model.compile(loss='mse', optimizer='rmsprop', metrics=['mae',r2])
#因为我们这是预测房价，一个线性回归预测，所以loss跟分类是不一样的。
#我们这里要用mean_squared_error 可简写成mse optimizer
#我这里用rmsprop大家可以用adam其它梯度下降优化函数，
#正确率我们这里用mae平均绝对误差和我们自定义的r2

#训练神经网络
# history = model.fit(train_x, train_y, batch_size=batch_size, epochs=training_epochs,  validation_data=(test_x, test_y))
#history = model.fit(train_x, train_y, batch_size=batch_size, epochs=training_epochs, validation_split=0.3)
history = model.fit(train_x, train_y, batch_size=batch_size, epochs=training_epochs)

pred_test_y = model.predict(test_x)
print(pred_test_y)

from sklearn.metrics import r2_score
pred_acc = r2_score(test_y, pred_test_y)
print('pred_acc',pred_acc)

#绘图
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置图形大小
plt.figure(figsize=(8, 4), dpi=80)
plt.plot(range(len(test_y)), test_y, ls='-.',lw=2,c='r',label='真实值')
plt.plot(range(len(pred_test_y)), pred_test_y, ls='-',lw=2,c='b',label='预测值')

# 绘制网格
plt.grid(alpha=0.4, linestyle=':')
plt.legend()
plt.xlabel('number') #设置x轴的标签文本
plt.ylabel('房价') #设置y轴的标签文本

# 展示
plt.show()