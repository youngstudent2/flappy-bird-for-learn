import flappybird as fb
import random
import time

import sklearn
class GeneticBrain(fb.Brain):
    def __init__(self):
        pass
    def decideFlap(self,params):
        pass
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
            
            new
    
    
    
def train():
    brain = HappyBrain()
    i = 0
    g = fb.FlappyBirdGame(60,1,[brain])
    while i < 5:   
        g.run()
        g.reset(1,[brain])
        i += 1
    
if __name__ == '__main__':
    train()