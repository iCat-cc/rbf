
#————————————————
#来源：小凌のBlog—Good Times|一个不咋地的博客https://blog.ling08.cn/
#原文链接：https://blog.ling08.cn/index.php/post-232.html
# -*- coding: utf-8 -*-
import math
import string
import matplotlib as mpl
############################################调用库（根据自己编程情况修改）
import numpy.matlib 
import numpy as np
np.seterr(divide='ignore',invalid='ignore')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd
import random
 
#生成区间[a,b]内的随机数
def random_number(a,b):
    return (b-a)*random.random()+a
 
#生成一个矩阵，大小为m*n,并且设置默认零矩阵
def makematrix(m, n, fill=0.0):
    a = []
    for i in range(m):
        a.append([fill]*n)
    return np.array(a)
#求取范数
def metrics(a): 
    return np.linalg.norm(a, axis=1, keepdims=True)
 
 
#构造三层BP网络架构
class BPNN:
    def __init__(self, num_in, num_hidden, num_out, num_p):
        #输入层，隐藏层，输出层的节点数
        self.num_in = num_in
        self.num_hidden = num_hidden + 1   #增加一个偏置结点
        self.num_out = num_out
        self.num_p = num_p
        #激活神经网络的所有节点（向量）
        self.active_in = np.array([-1.0]*self.num_in)
        self.active_hidden = np.array([-1.0]*self.num_hidden)
        self.active_out = np.array([1.0]*self.num_out)
 
        #创建初始化变量
        self.x_c=makematrix(self.num_p,self.num_hidden)
        self.G = makematrix(self.num_p,self.num_hidden)
        self.ei = makematrix(self.num_p,1)#所有样本误差信号
        self.i=0
        
        #创建权重矩阵以及其他变量
        self.C=makematrix(self.num_in,self.active_hidden)
        self.S=makematrix(self.num_hidden,1)
        self.wight_out = makematrix(self.num_hidden, self.num_out)
        
        #初始化权值以及其他变量
        for i in range(self.num_in):
            for j in range(self.num_hidden):
                self.C[i][j]=random_number(-1, 1)
        for i in range(self.num_hidden):
            for j in range(self.num_out):
                self.wight_out[i][j] = random_number(0.01, 0.2)
            self.S[i]=random_number(-1, 1)
        #初始化偏差
        for j in range(self.num_out):
            self.wight_out[0][j] = 0.5
  
 
 
    #基函数（高斯函数）sigmoid_hidden()
    def sigmoid_hidden(self,x):
        r=metrics(x-self.C.T)**2
        return np.exp(-1*r/(2*self.S**2))
 
    #函数sigmoid_out()
    def sigmoid_out(self,x):
        return np.dot(x.T,self.wight_out)
 
    #信号正向传播
    def update(self, inputs):
        if len(inputs) != self.num_in:
            raise ValueError('与输入层节点数不符')
            
        #数据输入输入层
        self.active_in=inputs
        #数据在隐藏层的处理
        self.active_hidden=self.sigmoid_hidden(self.active_in)   #active_hidden[]是处理完输入数据之后存储，作为输出层的输入数据
        self.active_hidden[0]=-1
        #数据在输出层的处理
        self.active_out = self.sigmoid_out(self.active_hidden)   #与上同理
        #print(self.active_out)
        return self.active_out
 
    def error_ei(self,targets,inputs):
         #误差
        if self.i==self.num_p:
            self.i=0
        self.ei[self.i]=targets-self.update(inputs)
        self.x_c[self.i]=inputs-self.C
        self.G[self.i]=self.active_hidden.T
        self.i=self.i+1
        
    #反向传播
    def errorbackpropagate(self,lr):   #lr是学习率
        self.error=(1/2)*(np.dot(makematrix(1,self.num_p,1),self.ei**2)) 
        self.C=self.C+lr*((self.wight_out/(self.S**2)).T)*np.dot(self.ei.T,self.G*self.x_c)
        self.S=self.S+lr*(self.wight_out/(self.S**3))*(np.dot(self.ei.T,self.G*(self.x_c)**2).T)
        self.wight_out=self.wight_out+lr*np.dot(self.ei.T,self.G).T 
        return self.error
 
    #测试
    def test(self, patterns):
        t=np.array([0.0]*21)
        a=0
        for i in patterns:
            print(i[0:self.num_in], '->', self.update(i[0:self.num_in]))
            t[a]=self.update(i[0:self.num_in])
            a+=1
        return t
 
    #权重
    def weights(self):
        print(">>>S")
        print(self.S)
        print(">>>C")
        print(self.C)
        print(">>>权值")
        print(self.wight_out)
    #训练
    def train(self, pattern, itera=1000, lr = 0.01):#训练1000次，学习率为0.01
        for i in range(itera):
            for j in pattern:
                inputs = j[0:self.num_in]
                targets = j[self.num_in]
                self.error_ei(targets,inputs)
            error = self.errorbackpropagate(lr)
            #print("!!!!!!!!!",(self.G*self.x_c))
            if i % 100 == 0:
                print('########################误差 %-.5f######################第%d次迭代' %(error,i))
 
#实例
X=np.arange(-1,1.1,0.1)
D=[-0.96, -0.577, -0.0729, 0.377, 0.641, 0.66, 0.461, 0.1336, -0.201, -0.434, -0.5, -0.393, -0.1647, 0.0988, 0.3072, 0.396, 0.3449, 0.1816, -0.0312, -0.2183, -0.3201]
patt=np.array(list(zip(X,D)))
#print(patt)
    #创建神经网络，1个输入节点，10个隐藏层节点，1个输出层节点,21个样本
n = BPNN(1, 10, 1, 21)
    #训练神经网络
n.train(patt)
    #测试神经网络
patt2=n.test(patt)
    #查阅权重值
n.weights() 
    #绘制图形
plt.plot(X,D)
plt.plot(X,patt2)
plt.show()
