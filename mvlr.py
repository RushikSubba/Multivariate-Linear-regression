#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
    Multivariate Linear Regression
    ---with lasso and ridge regularization
    ---with Z-Score and min-max normalization
    ---plotted contour of theta(1) and theta(2) with respect to cost function
"""

"""
    Command Line Input :
        python3 mvlr.py [filename] [number of rows to be skipped (default=1)] [delimiter (default=space)]
"""

import matplotlib.pyplot as plt
import numpy as np
import math, sys

"""
    Squared Error
        x      : training data
        y      : outputs
        th     : theta 
        lb     : lambda value for regularization
        reg    : regularization function

"""
def cost(x, y, th, lb = 0, reg = None):
    diff = x @ th - y
    return np.squeeze((1 / (2 * x.shape[0])) * diff.transpose() @ diff + (0 if reg == None else reg(lb, th)))

"""
    Gradient Descent
        x      : training data
        y      : outputs
        th     : theta
        iters  : no of iterations
        alpha  : learning rate
        lb     : lambda value for regularization
        reg    : regularization function
        regder : regularization function derivative
"""
def gr_dsc(x, y, th, iters = 10000, alpha = 0.001, lb = 0, reg = None, regder = None):
    iter_ = iters
    while(iters != 0) :
        th = th - alpha*(x.transpose() @ (x @ th - y) + (0 if regder == None else regder(lb, th)))
        iters -= 1
    print("After " + str(iter_) + " iterations with alpha = " + str(alpha) + ",Cost = " + str(cost(x, y, th, lb, reg)))
    return th

"""
    Lasso regularization
        th : theta
        lb : lambda value for regularization
"""
def lasso_reg(lb, th):
    return abs(th).sum() * lb
"""
    Lasso regularization derivative
        th : theta
        lb : lambda value for regularization
"""
def lasso_regder(lb, th):
    return lb * th / abs(th)

"""
    Ridge regularization
        th : theta
        lb : lambda value for regularization
"""
def ridge_reg(lb, th):
    return (lb / 2) * np.squeeze(th.transpose() @ th)

"""
    Ridge regularization derivative
        th : theta
        lb : lambda value for regularization
"""
def ridge_regder(lb, th):
    return lb * th

"""
    Plots Cost vs theta(i) for all theta
        x      : training data
        y      : outputs
        th     : theta
"""
def plotter(x, y, th):
    ti = np.linspace(-10, 10, num=1000)
    for i in range(th.shape[0]):
        th1 = np.copy(th)
        jv = []
        for j in ti:
            th1[i][0] = j
            jv.append(cost(x, y, th1))
        print(min(jv))
        plt.plot(ti, jv)
        plt.show()

"""
    Contour plots
        x      : training data
        y      : outputs
        th     : theta
"""
def contour(x, y, th):
    th1 = np.linspace(-1000, 1000, num=1000)
    th2 = np.linspace(-1000, 1000, num=1000)
    TH1, TH2 = np.meshgrid(th1, th2)
    
    th_copy = th.copy()
    jv = np.zeros((len(th1), len(th2)))
    for i in range(0,len(th1)):
        th_copy[1][0] = th1[i]
        for j in range(0, len(th2)):
            th_copy[2][0] = th2[j]
            jv[i][j] = cost(x, y, th_copy)
    
    plt.contour(TH1, TH2, jv)
    plt.show()
            
    
"""
    Error calculation
        x      : testing data
        y      : outputs
        th     : theta 
"""
def error_calculations(x, y, th):
    diff = x @ th - y
    sum_of_squared_errors = diff.transpose() @ diff
    Eabs = sum(np.absolute(diff)) / x.shape[0]
    Erms = math.sqrt(sum_of_squared_errors[0][0]) / x.shape[0]
    print("Eabs = {}".format(Eabs[0]))
    print("Erms = {}".format(Erms))

"""
    Min-Max normalization
        x      : non normalized data
        y      : non normalized output
        data   : data to be normalized 
"""     
def min_max_norm(x, y, x_normalized, y_normalized, data):
    rows = data.shape[0]
    columns = data.shape[1]
    
    mins = data.min(axis=0)
    maxs = data.max(axis=0)

    for i in range(columns):
        if i == columns - 1:
            for j in range(rows):
                y_normalized[j][0] = (data[j][i] - mins[i])/(maxs[i] - mins[i])
                y[j][0] = data[j][i]
        else :
            for j in range(rows):
                x_normalized[j][i + 1] = (data[j][i] - mins[i])/(maxs[i] - mins[i])
                x[j][i + 1] = data[j][i]

"""
    Z-Score Normalization
        x      : non normalized data
        y      : non normalized output
        data   : data to be normalized
"""
def z_score_norm(x, y, x_normalized, y_normalized, data):
    rows = data.shape[0]
    columns = data.shape[1]
    
    means = data.mean(axis = 0)
    std = data.std(axis = 0)
    
    for i in range(columns):
        if i == columns - 1:
            for j in range(rows):
                y_normalized[j][0] = (data[j][i] - means[i])/(std[i])
                y[j][0] = data[j][i]
        else :
            for j in range(rows):
                x_normalized[j][i + 1] = (data[j][i] - means[i])/(std[i])
                x[j][i + 1] = data[j][i]

#Command line arguments
argList = sys.argv 
f_name = argList[1]
delim = " "
skp_head = 1

if(len(argList) > 2):
    skp_head = int(argList[2])
if(len(argList) > 3):
    delim = argList[3]

#Initializing the input data
data = np.genfromtxt(f_name, delimiter=delim, skip_header=skp_head)
rows = data.shape[0]
columns = data.shape[1]

#Initializing the X, Y and Theta
th = np.ones((columns, 1))
x = np.ones((rows, columns))
y = np.zeros((rows, 1))
x_normalized = np.ones((rows, columns))
y_normalized = np.zeros((rows, 1))

#Normalize
opt = int(input("Enter 1 for Min-Max norm, 2 for Z-Score : "))
if(opt == 1):
    min_max_norm(x, y, x_normalized, y_normalized, data)
elif(opt == 2):
    z_score_norm(x, y, x_normalized, y_normalized, data)

#Splitting into learning and testing sets
x_learning = x_normalized[:int(0.8*rows) + 1,:]
x_testing = x_normalized[int(0.8*rows) + 1:,:]
y_learning = y_normalized[:int(0.8*rows) + 1,:]
y_testing = y_normalized[int(0.8*rows) + 1:,:]
th_copy = th.copy()

#Produce Contours
print("Normalized countour : ")
print("Close the contour window to continue...")
contour(x_normalized, y_normalized, th_copy)
print("Non-Normalized countour : ")
print("Close the contour window to continue...")
contour(x, y, th_copy)

reg = None
regder = None
opt = int(input("Enter 1 for lasso regularization, 2 for ridge regularizations, 3 for none "))
if(opt == 1):
    reg = lasso_reg
    reg_der = lasso_regder
elif(opt == 2):
    reg = ridge_reg
    reg_der = ridge_regder

th_copy = gr_dsc(x_learning, y_learning, th_copy, iters = 10000, alpha = 0.002, lb = 0.01, reg = reg, regder = regder)
error_calculations(x_testing, y_testing, th_copy)


# In[ ]:




