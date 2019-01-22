#!/usr/bin/env python
# coding: utf-8

# In[4]:


from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn import linear_model
import statistics


# In[1]:


#delete rows or colums with nan
# del_nan(df, col = False) - delete rows, if include nan, df - dataframe
def nan_detect(df):
    nan_pos = []
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            try:
                if  pd.isna(df.iloc[i,j]):             
                        nan_pos.append((i,j))
            except:
                pass
    return nan_pos

def del_nan(df, col = False):
    if col:
        df = df.drop(df.columns[[i[1] for i in nan_detect(df) ]], axis=1)

    else:
        df = df.drop(df.index[[i[0] for i in nan_detect(df) ]], axis = 0)

    return df


# In[ ]:


# replace nan with average, mode or mediana
# amm(a, n = 1, b = 'avarage'), n - number of column, a - dataframe
def amm(a, n = 1, b = 'avarage'):
    
    if b == 'avarage':
        
        s = 0    
        k = len(a)
    
        for i in range(len(a)): 
            if a[i][n] == a[i][n]:
                s = s + a[i][n]
            else:
                k -= 1        
        
        for i in range (len(a)):
            if a[i][n] != a[i][n]:
                a[i][n] = s/k
        return (a)
    
    elif b == 'mediana':
        
        l = []
        for i in range(len(a)):
            k = len(a)
            if a[i][n] == a[i][n]:            
                l.append(a[i][n])            
            else:
                k -= 1
        
        l.sort()
            
        for i in range (len(a)):
            if a[i][n] != a[i][n]:
                a[i][n] = l[len(l)//2]    
        return (a)
    
    elif b == 'moda':    
        
        l = []
        for i in range(len(a)):        
            k = len(a)        
            if a[i][n] == a[i][n]:            
                l.append(a[i][n])             
            else:
                k -= 1          
            
        for i in range (len(a)):
            if a[i][n] != a[i][n]:
                a[i][n] = max(l, key=l.count)    
        return (a)
    
    else:
        print ('please use avarage, moda or mediana')
        
        
        


# In[ ]:


# replace nan with linear regression result
#LR(a, n = 0), n - number of column for prediction, k - number of colum dependence// more easy linear regression


def LR(a, n = 0, k = 1):
    
    a1 = a
    
    #functions for del nan
    def nan_detect(df):
        nan_pos = []
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                try:
                    if  pd.isna(df.iloc[i,j]):             
                        nan_pos.append((i,j))
                except:
                    pass
        return nan_pos
    
    def del_nan(df, col = False):
        if col:
            df = df.drop(df.columns[[i[1] for i in nan_detect(df) ]], axis=1)

        else:
            df = df.drop(df.index[[i[0] for i in nan_detect(df) ]], axis = 0)

        return df
    
    
    # Linear regression
    
    a = pd.DataFrame(a)
    a = del_nan(a)
    y = np.array(a[n])    
    X = np.array(a[k])
    
        
    #X = X.T 
    X = np.c_[X, np.ones(X.shape[0])] 
    beta_hat = np.linalg.lstsq(X,y)[0]
    
    beta_hat1 = beta_hat[-1]
    beta_hat = beta_hat[:-1]

  
    def listsum(numList):
        if len(numList) == 1:
            return numList[0]
        else:
            return numList[0] + listsum(numList[1:])  
        
    # new coeficients    
    
    for i in range (len(a1)):
        f = []
        
        if a1[i][n] != a1[i][n]:    
                              
            f.append(a1[i][k])                          
                     
            c = np.multiply(np.array(f),beta_hat1)   
                         
            a1[i][n] = (listsum(c) + beta_hat[-1])
            
    return a1
            


# In[17]:


# replace nan with linear regression result
#LR(a, n = 0), n - number of column // more difficult linear regression - take all columns as dependensies, but they can not include nan variables

def LR1(a, n = 0):
    
    a1 = a
    
    #functions for del nan
    def nan_detect(df):
        nan_pos = []
        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                try:
                    if  pd.isna(df.iloc[i,j]):             
                        nan_pos.append((i,j))
                except:
                    pass
        return nan_pos
    
    def del_nan(df, col = False):
        if col:
            df = df.drop(df.columns[[i[1] for i in nan_detect(df) ]], axis=1)

        else:
            df = df.drop(df.index[[i[0] for i in nan_detect(df) ]], axis = 0)

        return df
    
    
    # Linear regression
    
    a = pd.DataFrame(a)
    a = del_nan(a)
    y = np.array(a[n])
    b = a[:]
    del b[n]
    X = np.array(b[:])
    #X = X.T 
    X = np.c_[X, np.ones(X.shape[0])] 
    beta_hat = np.linalg.lstsq(X,y)[0]
    
    beta_hat1 = beta_hat[:-1]
    
    #list sum
    
    def listsum(numList):
        if len(numList) == 1:
            return numList[0]
        else:
            return numList[0] + listsum(numList[1:])  
        
    # new coeficients    
        
    for i in range (len(a1)):
        f = []
        if a1[i][n] != a1[i][n]: 
        
            for j in range (len (a1[i])):        
            
                f.append(a1[i][j])
                             
            del f[n]
         
            c = np.multiply(np.array(f),beta_hat1)   
              
            a1[i][n] = (listsum(c) + beta_hat[-1])
    return a1
        


# In[ ]:


#standardizes your data
#st(a, n = 0), a - data, n - number of column


def st(a, n = 0):
    
    l = []
    for i in range(len(a)):        
        if a[i][n] == a[i][n]:            
            l.append(a[i][n])            
    s = statistics.stdev(l) 
    m = statistics.mean(l) 
    
    
    for i in range (len(a)):
        if a[i][n] == a[i][n]:
            a[i][n] = (a[i][n] - m)/s
        else:
            print ("your data include nan")
    return (a)


# In[2]:


# scales values in your data
#msh(a, n = 0) - a your data, n - number of column


def msh(a, n = 0):
    
    l = []
    for i in range(len(a)):        
        if a[i][n] == a[i][n]:            
            l.append(a[i][n])            
        
    
    for i in range (len(a)):
        if a[i][n] == a[i][n]:
            a[i][n] = (a[i][n] - min(l))/(max(l)-min(l))
        else:
            print ("your data include nan")
    return (a)
       


# In[ ]:


#KNN

def get_nan_list(column):
    from math import isnan
    nan_list = []
    for i in range(len(column)):
        if isnan(column[i]): 
            nan_list.append(i)
    return nan_list

def get_train_test(data, column_name, nan_list):
    all_list = list(range(0, data.shape[0]))
    not_nan_list = list(set(all_list)-set(nan_list))
    data_predict = data.iloc[nan_list, :]
    data_train = data.iloc[not_nan_list, :]
    return data_train, data_predict

def get_columns(data, column_name):
    columns = list(data.columns)
    del columns[columns.index(column_name)]
    return columns
    
def euclidean_distance(node1, node2, shape):
    import math
    distance = 0
    for x in shape:
        distance += pow((node1[x] - node2[x]), 2)
    return math.sqrt(distance)

def get_neighbors(data, column_name, amount_k):
    if data[column_name].dtype=='object':
        raise ValueError
    nan_list = get_nan_list(data[column_name])
    all_list = list(range(0, data.shape[0]))
    not_nan_list = list(set(all_list)-set(nan_list))
    if len(nan_list)==0:
        raise ValueError
    data_train, data_predict = get_train_test(data, column_name, nan_list)        
    columns = get_columns(data, column_name)
    all_neighbors = []
    for y in nan_list:
        distances = []
        for x in not_nan_list:
            dist = euclidean_distance(
                data_predict.loc[y, columns], 
                data_train.loc[x, columns], 
                columns)
            distances.append((x, dist))
        distances.sort(key=lambda x:x[1])
        all_neighbors.append(distances[:amount_k])  
        print(y)  
    return all_neighbors

def get_target(neighbors, target):
    sum_ = 0
    sum_d = 0
    for i in neighbors:
        if i[1]==0:
            return target[i[0]]
        sum_d +=1/i[1]
        sum_+=1/i[1]*target[i[0]]
    return sum_/sum_d



def replace_knn(data, column_name, amount_k):
    if data[column_name].dtype=='object':
        raise ValueError
    nan_list = get_nan_list(data[column_name])
    if len(nan_list)==0:
        raise ValueError
    predictions = []
    neighbors = get_neighbors(data, column_name, amount_k)
    for i in neighbors:
        predictions.append(get_target(i, data[column_name]))
    data_=data.copy(deep=True)
    for i in range(len(nan_list)):
        data_.loc[nan_list[i], column_name]=predictions[i]
    return data_

