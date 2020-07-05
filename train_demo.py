import flappybird as fb
import pygame
from pygame.locals import *
import random
import time
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
    brain = HappyBrain()
    i = 0
    g = fb.FlappyBirdGame(60,5,[brain]*5)
    while i < 5:   
        g.run()
        g.reset(5,[brain]*5)
        i+=1
    
if __name__ == '__main__':
    train()