import flappybird as fb
import pygame
from pygame.locals import *
import random
import time
class PlayerBrain(fb.Brain): # 玩家大脑
    
    def decideFlap(self,params):
        #print(params)
        for e in pygame.event.get(): 
            if e.type == MOUSEBUTTONDOWN or (e.type == KEYDOWN and
                    e.key in (K_UP, K_RETURN, K_SPACE)):
                return True
        return False
class HappyBrain(fb.Brain):
    def __init__(self):
        random.seed(2000)
    def decideFlap(self,params):
        #print(params)
        if params['height'] < 40:
            return False
        r = random.randint(0,1000)       
        return r > 940   




def train():
    brain = HappyBrain()
    g = fb.FlappyBirdGame(60,1,[brain])
    g.run()

    
if __name__ == '__main__':
    train()