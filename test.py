FPS = 60

class A():
    def __init__(self,fps = FPS):
        global FPS
        FPS = fps
        print("in class:")
        print(FPS)
class B():
    def check(self):
        print("b check")
        print(FPS)
b = B()
b.check()
a = A(30)
print(FPS)
b.check()
a = A()
print(FPS)
b.check()
a = A(70)
print(FPS)
b.check()
