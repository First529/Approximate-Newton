import numpy as np

class Ridge_Logistic_Regression():
    
    def __init__(self, lambd=0.1):
        self.lambd = lambd
        
    def loss(self, A, b, x):
        return np.sum(np.log(1 + np.exp(-b * (A @ x)))) + 0.5*self.lambd*(x**2).sum()
        
    def gradient(self, A, b, x):
        return -A.T @ (b/(1 + np.exp(b * (A @ x))))
        
    def hessian(self, A, b, x):
        '''
        Computation in matrix form to find augmented diagonal matrix 
        '''
        return (1/(1 + np.exp(b * (A @ x)))) * (1/(1 + np.exp(-b * (A @ x))))
    
    def condition_num(self, A, b, x):
        _, d = A.shape
        return np.linalg.cond((A.T * self.hessian(A, b, x)) @ (A) + self.lambd * np.eye(d))
