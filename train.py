import flappybird as fb
import random
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import numpy as np
import copy
SCALE_FACTOR = 200
class GeneticBrain(fb.Brain):
    def __init__(self,n_input,n_hidden):
        '''
        self.model = Sequential()
        
        self.model.add(Dense(n_hidden,activation='sigmoid',input_shape=(n_input,)))
        self.model.add(Dense(1,activation='sigmoid'))
        #print(self.getModel())
        '''
        self.model = NeuralNetwork([n_input,n_hidden],'logistic')
    def decideFlap(self,params):
        #print(params)
        distance = params['distance'] + params['pipeWidth'] 
        deltaHeight = (params['bottomPipeHeight'] + params['topPipeHeight'])/2 - params['height']
        velY = params['velY']
        data = [distance * SCALE_FACTOR, deltaHeight * SCALE_FACTOR]
        pred = self.model.predict(data)
        #print(pred)
        return pred[0] > 0.5
    
    def getModel(self):
        return self.model.getWeights()
        
    def setModel(self,weights):
        self.model.setWeights(weights)
        return True

class GeneticAlgorithm():
    
    def __init__(self,max_units,top_units):
        self.max_units = max_units
        self.top_units = top_units
        
        if max_units < top_units:
            self.top_units = max_units
        self.population = []

        self.best_brain = None

    def reset(self):
        self.iteration = 1
        self.mutateRate = 1
        self.best_population = 0
        self.best_fitness = 0
        self.best_score = 0
    
    def createPopulation(self):
        self.population = []
        for i in range(self.max_units):
            
            newUnit = GeneticBrain(2,6)
            newUnit.index = i
            newUnit.fitness = 0
            newUnit.score = 0
            newUnit.isWinner = False
            
            self.population.append(newUnit)
        return self.population
    def evolvePopulation(self,results):
        winners = self.selection(results)

        for w in winners:
            print("%d: fitness = %f score = %d" %(w.index,w.fitness,w.score))
        
        if self.mutateRate == 1 and winners[0].fitness < 0:
            # all is bad
            # create another population
            print("recreate popultation")
            return self.createPopulation()
        else:
            self.mutateRate = 0.2
        
        if winners[0].fitness > self.best_fitness:
            self.best_fitness = winners[0].fitness
            self.best_score = winners[0].score
            winners[0].model.save('best.h5')
        
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
                offspring = winners[0].getModel()
            
            offspring = self.mutation(offspring)
            
            newUnit = self.population[i]
            newUnit.setModel(offspring)
            newUnit.score = 0
            newUnit.isWinner = False
            
        
        
        return self.population
            
    def selection(self,results):
        for i in range(self.top_units):
            self.population[results[i].index].isWinner = True
        return results[:self.top_units]    
        
    
    def crossOver(self,parantA,parantB):
        length = np.size(parantA[1],0)
        cutPoint = random.randint(0,length-1)
        
        for i in range(cutPoint,length):
            tmp = parantA[1][0][i]
            parantA[1][0][i] = parantB[1][0][i]
            parantB[1][0][i] = tmp
        
        
        if random.randint(0,1):
            return parantA
        else:
            return parantB
    def mutation(self,offspring):

        for i in offspring[1]:
            for bias in i:
                bias = self.mutate(bias)
        for i in offspring[0]:
            for weight in i:
                weight = self.mutate(weight)
        return offspring
    def mutate(self,gene):
        if random.random() < self.mutateRate:
            mutateFactor = 1 + (random.random() - 0.5) * 3 + (random.random() - 0.5)
            gene *= mutateFactor
        return gene
    
    def getRandomUnit(self,array):
        return array[random.randint(0,len(array)-1)]
    
    def normalize(self,value,maxValue):
        if value < -maxValue: value = -maxValue
        elif value > maxValue: value = maxValue
        return value/maxValue
    
    def saveBestBird(self):
        pass
    
import pygame    
class PlayerBrain(fb.Brain): # 玩家大脑
    
    def decideFlap(self,params):
        #print(params)
        return params['playerClick']
class HappyBrain(fb.Brain):
    def __init__(self):
        random.seed(2000)
    def decideFlap(self,params):
        #print(params)
        pygame.event.get()
        if params['height'] < 40:
            return False
        r = random.randint(0,1000)       
        return r > 940      
    
    
    
    
def train():
    bird_num = 10
    
    GA = GeneticAlgorithm(bird_num,4)
    GA.reset()
    brains = GA.createPopulation()
    
    #brains = [HappyBrain()] * bird_num

    g = fb.FlappyBirdGame(30,bird_num,brains)
    train_time = 200
    for i in range(train_time):

        g.run()
        results = g.result()
        
        print("Generation %d:" %(i))
        
        sorted_brains = []
        for r in results[::-1]:
            b = r[0].brain
            b.fitness = (r[1]['score']) * r[1]['interval'] - r[1]['distance']
            b.score = r[1]['score']
            sorted_brains.append(b)
        
        brains = GA.evolvePopulation(sorted_brains)
        print("best score = %d   best fitness = %d" % (GA.best_score,GA.best_fitness))
        g.reset(bird_num,brains)
    GA.saveBestBird()
    print("GA end!")
    
    
from simpleNeuralNetwork import NeuralNetwork


class simpleNNBrain(fb.Brain):
    def __init__(self):
        self.model = NeuralNetwork([2,6,1],'logistic')
        print(self.model.getWeights())
    def decideFlap(self,params):
        distance = params['distance'] + params['pipeWidth'] 
        deltaHeight = (params['bottomPipeHeight'] + params['topPipeHeight'])/2 - params['height']
        velY = params['velY']
        data = [distance * SCALE_FACTOR, deltaHeight * SCALE_FACTOR]
        pred = self.model.predict(data)
        #print(pred)
        print(pred)     
        return pred[0] > 0.5   
    
def train_test():
    bird_num = 10
    brains = []
    for i in range(bird_num):
        brains.append(simpleNNBrain())
    
    g = fb.FlappyBirdGame(30,bird_num,brains)
    for i in range(10):
        g.run()
        result = g.result()
        brains = []
        for i in range(bird_num):
            brains.append(simpleNNBrain())
        g.reset(10,brains)
if __name__ == '__main__':
    train()
    