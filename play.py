from turtle import color
import numpy as np
import random
from random import randint
import matplotlib.pyplot as plt
import numpy as np
class Perceptron:
    def __init__(self, N):
        #random.seed(0)
        self.N = N
        #set a random line by x_a,y_a,x_b,y_b to divide the data into two categories
        x_a,y_a,x_b,y_b = [random.uniform(-1, 1) for i in range(4)]
        w2 = 1 / (x_b * y_a - x_a * y_b)
        w1 = -1 * (y_a - y_b) / (x_a - x_b) * w2
        #weight
        self.W = np.array([1, w1, w2])
        #generate random data points
        self.X = self.generate_data()
        #label random data points by the random line
        self.labels = np.sign(self.W.dot(self.X))

    def generate_data(self):
        #random.seed(0)
        X = np.empty([3,self.N])
        for n in range(self.N):
            x1, x2 = [random.uniform(-1, 1) for i in range(2)]
            X[0][n] = 1
            X[1][n] = x1
            X[2][n] = x2	
        return X

    def guessY(self, x):
        b = self.W[0]
        w1 = self.W[1]
        w2 = self.W[2]
        return (-(b/ w2) / (b/w1))*x + (-b / w2)

    def plot(self):
        x = np.arange(-1.,1.1,0.1)
        y = -1*(1+self.W[1]*x)/self.W[2] 
        
        plt.figure(figsize=(5,5))
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.plot(x,y)
        
        i = 0
        while i < self.N:
            if self.labels[i] > 0:
                plt.plot(self.X[1][i],self.X[2][i],'ro')
            else:
                plt.plot(self.X[1][i],self.X[2][i],'bo')
            i+=1
        #plt.savefig('test1.png')
        #plt.plot([-1,1], [y1, y2], 'bo', linestyle ='--')
        plt.show()
    def perceptronLearningAlgorithm(self):
        rand_weight = np.array([1.,0.,0.])
        x1 = np.arange(-1.,1.1,0.1)
        slope = -(1/self.W[2])/(1/self.W[1])  
        intercept = -self.W[0]/self.W[2]
        y1 = (slope*x1) + intercept
        plt.plot(x1,y1)

        iteration = 1000
        correct = False
        i = 0
        avg_disagreement_p = 0
        while (correct != True):
           #pick a point randomly from all points
            if i == 0:
                random = randint(0,self.N-1)
                rand_x = self.X[:,random]
                for k in rand_x:
                    rand_weight += k*self.labels[random]  
            else:
                error = self.error(rand_weight)
                if len( error ) == 0:
                    print (str(i)+" iterations before convergence!")
                    correct = True
                    j = 0
                    while j < self.N:
                        if self.labels[j] > 0:
                            plt.plot(self.X[1][j],self.X[2][j],'ro')
                        else:
                            plt.plot(self.X[1][j],self.X[2][j],'bo')
                        j+=1
                    x = np.arange(-1.,1.1,0.1)
                    slope = -(1/rand_weight[2])/(1/rand_weight[1])  
                    intercept = -rand_weight[0]/rand_weight[2]
                    y = (slope*x) + intercept
                    plt.plot(x,y,color='r')
                    plt.xlim(-1,1)
                    plt.ylim(-1,1)
                    plt.xlabel('x1')
                    plt.ylabel('x2')
                    plt.show()
                    break
                random = randint(0,len(error)-1)
                rand_x = self.X[:,error[random]]
                rand_weight += rand_x*self.labels[error[random]]            
            i+=1 
        return i

    def error(self, rand_weigth):
        error = []
        i = 0
        labels = np.sign(rand_weigth.dot(self.X))
        while i < self.N:
            if labels[i] != self.labels[i]:
                error.append(i)
            i+=1
        return error 
    def disagreement_p(self, rand_w):
        #test disagreement by random 100 points
        random_pt_test = 100
        disagreement = 0
        for i in range(random_pt_test):
            x = np.array([1, random.uniform(-1,1),random.uniform(-1,1)])
            if np.dot(rand_w, x) != np.dot(self.W, x):
                disagreement += 1
        #outputs the probability of disagreement
        return float(disagreement)/float(random_pt_test) 
    
            
#start PLA
run = 100
avg_iter = 0

# p = Perceptron(100)
# p.perceptronLearningAlgorithm()

convergence = []

for i in range(run):
    iteration = Perceptron(1000).perceptronLearningAlgorithm()
    avg_iter += iteration
    convergence.append(iteration)

import math 
def roundup(x):
    return int(math.ceil(x/100.0))*100
avg_iter = float(avg_iter)/float(run)
print ("after 1000 runs, the avg num of iteration is "+str(avg_iter)+"!")
print(convergence)
print(max(convergence))
print(roundup(max(convergence)))

total =roundup(max(convergence))
total = int(total /100)
print(total)
bins = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000]
binwidth = (max(convergence) - min(convergence)) / 15
plt.figure()
plt.hist(convergence, bins=np.arange(min(convergence), max(convergence) + binwidth, binwidth))
plt.hist(convergence, bins)
plt.show()
