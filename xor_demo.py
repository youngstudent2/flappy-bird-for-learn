from simpleNeuralNetwork import NeuralNetwork
import numpy as np
  
nn = NeuralNetwork([2,2,1],'tanh')
  
# 验证异或逻辑
x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])
nn.fit(x,y)
for i in [0,0],[0,1],[1,0],[1,1]:
    print(i,nn.predict(i))

print(nn.getWeights())
