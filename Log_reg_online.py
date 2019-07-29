# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:38:51 2019

@author: yashd
"""

import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:

    def __init__(self, alpha=0.001, epochs=100000):
        self.alpha = alpha
        self.epochs = epochs
        self.loss_threshold = 0.001
        self.weights = np.ones(3)
        self.current_loss = float('inf')
        self.previous_loss = float('inf')
        self.training_converged = False
        self.iteration_count = 0
        self.total_cost = []
        self.tot_iter = []
        self.norm_current_loss = []
        self.norm_grad=[]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cross_entropy(self, predicted, actual):
        self.cost = (-actual * np.log(predicted) - (1 - actual) * np.log(1 - predicted)).mean()
        self.total_cost.append(self.cost)
        return self.cost

    def chunk_learn(self, X, y):
        net_val = np.dot(X, self.weights)
        prediction = self.sigmoid(net_val)
        self.gradient = np.dot(X.T, (prediction - y))
        self.weights -= self.alpha * self.gradient
        self.current_loss = self.cross_entropy(prediction, y)
        self.norm_grad.append(abs(self.gradient[0])+abs(self.gradient[1])+abs(self.gradient[2]))
        if self.current_loss < self.loss_threshold:
            print("total no. of iterations run: ", self.iteration_count + 1)
            self.training_converged = True
        
        self.norm_current_loss.append(abs(self.current_loss-self.previous_loss))
        self.previous_loss = self.current_loss
        self.iteration_count += 1
        self.store_iter = self.iteration_count
        self.tot_iter.append(self.store_iter)
    
        return self.norm_current_loss, self.tot_iter, self.norm_grad

    def fit(self, X, target):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)

        for epoch in range(self.epochs):
            for i in range(X.shape[0]):
                self.chunk_learn(X[i, :], target[i])
                if self.training_converged:
                    break

            if self.training_converged:
                break

    def predict(self, X):
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return np.round(self.sigmoid(np.dot(X, self.weights)))

def testdataviz(weights,x1,y1,a1,b1):
    slope = (-weights[0]/weights[2])/(weights[0]/weights[1])  
    intercept = -weights[0]/weights[2]  
    
    print('Equation of Separation Line : y = ',slope,'x + ',intercept)
    
    plt.scatter(x1, y1, marker='o')
    plt.axis('equal')
    plt.scatter(a1, b1, marker='x')
    plt.axis('equal')
    plt.title('Test Data Visualization')
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())     
    y_vals = slope * x_vals + intercept   
    plt.plot(x_vals, y_vals, '--')
    plt.show()    

def generate_data(mean, variance, count):
    return np.random.multivariate_normal(mean, variance, count)

def norm_gradient_graph(norm_grad,alpha):
    f, ax = plt.subplots(1, figsize=(5, 5))
    ax.set_title("Norm Gradient")
    ax.plot(range(0, len(norm_grad)), norm_grad, 'r', label=r'$\alpha = {0}$'.format(alpha)) 
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Gradient')
    ax.legend();

def calculateAccuracy(predicted_y, test_y):
    predicted_y = predicted_y.tolist()
    test_y = test_y.tolist()

    count = 0
    for i in range(len(predicted_y)):
        if predicted_y[i] == test_y[i]:
            count += 1

    return (count / len(predicted_y)) * 100

def cross_entropy_graph(total_cost,alpha):
    f, ax = plt.subplots(1, figsize=(5, 5))
    plt.figure(figsize=(8, 8))
    ax.set_title("Changes of Training Error")
    ax.plot(range(0, len(total_cost)), total_cost, 'r', label=r'$\alpha = {0}$'.format(alpha)) 
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Training Error')
    ax.legend();

if __name__ == '__main__':
    
    x1 = np.random.multivariate_normal([1, 0], [[1, 0.75], [0.75, 1]], 500)
    x2 = np.random.multivariate_normal([0, 1.5], [[1, 0.75], [0.75, 1]], 500)
    X_train = np.vstack((x1, x2)).astype(np.float32)
    Y_train = np.hstack((np.zeros(500), np.ones(500)))
    
    test_x1 = np.random.multivariate_normal([1, 0], [[1, 0.75], [0.75, 1]], 500)
    test_x2 = np.random.multivariate_normal([0, 1.5], [[1, 0.75], [0.75, 1]], 500)
    x_test = np.vstack((test_x1, test_x2)).astype(np.float32)
    y_test = np.hstack((np.zeros(500), np.ones(500)))
    
    max_epochs = 100000
    Acc = []   
    
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    print("\n\nLearning rate (Alpha): 1")
    LR = LogisticRegression(alpha=1, epochs=max_epochs)
    LR.fit(X_train, Y_train)
    predicted_y = LR.predict(x_test)
    accuracy = calculateAccuracy(predicted_y, y_test)
    Acc.append(accuracy)
    print("Accuracy: ", accuracy)
    print("Final Weights: ", LR.weights)
    norm_gradient_graph(LR.norm_grad,1)
    cross_entropy_graph(LR.total_cost,1)
    testdataviz(LR.weights,test_x1[:,0],test_x1[:,1],test_x2[:,0],test_x1[:,1])
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    
    print("\n\nLearning rate (Alpha): 0.1")
    LR = LogisticRegression(alpha=0.1, epochs=max_epochs)
    LR.fit(X_train, Y_train)
    predicted_y = LR.predict(x_test)
    accuracy = calculateAccuracy(predicted_y, y_test)
    Acc.append(accuracy)
    print("Accuracy: ", accuracy)
    print("Final Weights: ", LR.weights)
    norm_gradient_graph(LR.norm_grad,0.1)
    cross_entropy_graph(LR.total_cost,0.1)
    testdataviz(LR.weights,test_x1[:,0],test_x1[:,1],test_x2[:,0],test_x1[:,1])
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
   
    print("\n\nLearning rate (Alpha): 0.01")
    LR = LogisticRegression(alpha=0.01, epochs=max_epochs)
    LR.fit(X_train, Y_train)
    predicted_y = LR.predict(x_test)
    accuracy = calculateAccuracy(predicted_y, y_test)
    Acc.append(accuracy)
    print("Accuracy: ", accuracy)
    print("Final Weights: ", LR.weights)
    norm_gradient_graph(LR.norm_grad,0.01)
    cross_entropy_graph(LR.total_cost,0.01)
    testdataviz(LR.weights,test_x1[:,0],test_x1[:,1],test_x2[:,0],test_x1[:,1])
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    
    print("\n\nLearning rate (Alpha): 0.001")
    LR = LogisticRegression(alpha=0.001, epochs=max_epochs)
    LR.fit(X_train, Y_train)
    predicted_y = LR.predict(x_test)
    accuracy = calculateAccuracy(predicted_y, y_test)
    Acc.append(accuracy)
    print("Accuracy: ", accuracy)
    print("Final Weights: ", LR.weights)
    norm_gradient_graph(LR.norm_grad,0.001)
    cross_entropy_graph(LR.total_cost,0.001)
    testdataviz(LR.weights,test_x1[:,0],test_x1[:,1],test_x2[:,0],test_x1[:,1])
    print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
    

    