import numpy as np

# 定义两个sigmoid函数及其倒数
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1.0 - np.tanh(x)*np.tanh(x)

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

# 定义一个简单神经网络类
class NeuralNetwork:
    def __init__(self,layers,activation = 'tanh'):
        
        '''
        layers:定义神经网络各层神经元的个数，例如[10,10,3]表示3层神经网络，第一层10个神经元，以此类推
        activation:定义神经网络的激活函数，可选值（tanh,logistic)
        '''
        if activation == 'logistic':
            self.activation = logistic
            self.activation_deriv = logistic
        elif activation == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_derivative
        
        # 定义并初始化权重，怎么初始化？
        self.weights = []
        for i in range(1,len(layers) - 1):
            self.weights.append((2*np.random.random((layers[i-1] + 1,layers[i] + 1))-1)*0.25)
            self.weights.append((2*np.random.random((layers[i] + 1,layers[i+1]))-1)*0.25)
    
    # 训练    
    def fit(self,X,y,learning_rate = 0.2,epochs = 10000):
        X = np.atleast_2d(X)
        temp = np.ones([X.shape[0],X.shape[1] + 1])
        temp[:,0:-1] = X
        X = temp
        y = np.array(y)
        
        # 使用抽样梯度算法
        for k in range(epochs):
            # 从所有样本中，随机取一行
            i = np.random.randint(X.shape[0])
            a = [X[i]]
            
            # 完成正向的所有更新
            for l in range(len(self.weights)):
                a.append(self.activation(np.dot(a[l],self.weights[l])))
            
            # 计算输出层错误率
            error = y[i] - a[-1]
            deltas = [error*self.activation_deriv(a[-1])]
            
            # 计算隐藏层
            for l in range(len(a) -2,0,-1):
                deltas.append(deltas[-1].dot(self.weights[l].T)*self.activation_deriv(a[l]))
            deltas.reverse()
            
            # 反向更新权重
            for i in range(len(self.weights)):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                self.weights[i] += learning_rate * layer.T.dot(delta)
    
    # 预测，即正向更新过程                    
    def predict(self,x):
        x = np.array(x)
        temp = np.ones(x.shape[0] + 1)
        temp[0:-1] = x
        a= temp
        
        for l in range(0,len(self.weights)):
            a = self.activation(np.dot(a,self.weights[l]))
        return a
    def getWeights(self):
        return self.weights
    def setWeights(self,weights):
        self.weights = weights