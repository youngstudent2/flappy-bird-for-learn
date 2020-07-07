import flappybird as fb
import random
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
class GeneticBrain(fb.Brain):
    def __init__(self,n_input,n_hidden):
        self.model = Sequential()
        
        self.model.add(Dense(n_hidden,activation='relu',input_shape=(n_input,)))
        self.model.add(Dense(1))
        self.model.compile(loss='categorical_crossentropy', optimizer=SGD(), metrics=['accuracy'])
    def decideFlap(self,params):
        print(params)
        distance = params['distance']
        deltaHeight = (params['bottomPipeHeight'] + params['topPipeHeight'])/2 - params['height']
        velY = params['velY']
        data = np.array([[distance,deltaHeight,velY]])
        print(data.shape)
        pred = self.model.predict(data)
        return pred > 0.5
    def getModel(self):
        dense_layer = self.model.get_layer(index = 0)
        return dense_layer.get_weights()
    def updateModel(self,weights):
        dense_layer = self.model.get_layer(index = 0)
        dense_layer.set_weights(weights)
        return True

class GeneticAlgorithm():
    def __init__(self,max_units,top_units):
        self.max_units = max_units
        self.top_units = top_units
        
        if max_units < top_units:
            self.top_units = max_units
        
        self.population = []
        self.SCALE_FACTOR = 200

    def reset(self):
        self.iteration = 1
        self.mutateRate = 1
        self.best_population = 0
        self.best_fitness = 0
        self.best_score = 0
    
    def createPopulation(self):
        self.population = []
        for i in range(self.max_units):
            
            newUnit = GeneticBrain(3,6,1)
            newUnit.index = i
            newUnit.fitness = 0
            newUnit.score = 0
            newUnit.isWinner = False
            
            self.population.append(newUnit)
    
    def evolvePopulation(self):
        winners = self.selection()
        
        if self.mutateRate == 1 and winners[0].fitness < 0:
            # all is bad
            # create another population
            self.createPopulation()
        else:
            self.mutateRate = 0.2
        
        for i in range(self.top_units,self.max_units):
            if i == self.top_units:
                parantA = winners[0].getModel()
                parantB = winners[1].getModel()
                offspring = self.crossOver(parantA,parantB)
            elif i < self.max_units - 2:
                parantA = self.getRandomUnit(winners).getModel()
                parantB = self.getRandomUnit(winners).getModel()
                offspring = self.crossOver(parantA,parantB)
            else:
                offspring = self.getRandomUnit(winners).getModel()
            
            offspring = self.mutation(offspring)
            
            newUnit = self.population[i]
            newUnit.setModel(offspring)
            newUnit.score = 0
            newUnit.isWinner = False
        if winners[0].fitness > self.best_fitness:
            self.best_fitness = winners[0].fitness
            self.best_score = winners[0].score
        
        
            
    def selection(self,result):
        for i in range(self.top_units):
            self.population[result[i][0].brain.index].isWinner = True
        return result[:self.top_units]    
        
    
    def crossOver(parantA,parantB):
        length = np.size(parantA,0)
        cutPoint = random.randint(0,length-1)
        
        for i in range(cutPoint,length):
            tmp = parantA[1][i]
            parantA[1][i] = parantB[1][i]
            parantB[1][i] = tmp
        
        
        if random.randint(0,1):
            return parantA
        else:
            return parantB
    def mutation(offspring):
        length = np.size(offspring,0)
        for i in range(length):
            offspring[1][i] = self.mutate(offspring[1][i])
            offspring[0][i] = self.mutate(offspring[0][i])
        return offspring
    def mutate(gene):
        if random.random() < self.mutateRate:
            mutateFactor = 1 + (random.random() - 0.5) * 3 + (random.random() - 0.5)
            gene *= mutateFactor
        return gene
    
    def getRandomUnit(self,array):
        return array[random.randint(0,len(array)-1)]
    
    def normalize(value,maxValue):
        if value < -maxValue: value = -maxValue
        elif value > maxValue: value = maxValue
        return value/maxValue
def train():
    brain = GeneticBrain(3,6)
    i = 0
    g = fb.FlappyBirdGame(60,1,[brain])
    while i < 5:   
        g.run()
        g.reset(1,[brain])
        i += 1

if __name__ == '__main__':
    train()
    