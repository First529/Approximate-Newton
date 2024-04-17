import numpy as np

class f():
    
    def __init__(self, lambd=0.1):
        self.lambd = lambd
        
    def loss(self, A, b, x):
        return x.T@A@x+b.T@x
        
    def gradient(self, A, b, x):
        return 2*A.T@x+b
        
    def hessian(self, A, b, x):
        return 2*A
    
