
# coding: utf-8

# In[1]:

#######  Numpy   ########


# In[4]:

import numpy as np
a = [1,2,3,4]
b = np.array(a)
print b
type(b)  ##Type 
b.shape()
b.mean()


# In[21]:

c = [[1,2],[3,4],[3,4]]
d = np.array(c)
print d.max(axis = 0) # each colums, 最大值， 找维度0，也就是最后一个维度上的最大值，array([3, 4])

print d.max(axis = 1) # each row最大值,纵坐标改变，横坐标不变
d.mean(axis = 0)
d.mean(axis = 1)


# In[28]:

d.flatten()
# c 不行，'list' object has no attribute 'flatten'
np.ravel(c) ## ravel flatten都可以
np.ravel(d)


# In[34]:

e = np.ones((3,3), dtype = np.float) # 3*3 浮点类型2维数组，且初始化所有元素值为1
f = np.repeat(3,4)


# In[49]:

g = np.zeros((2,2,3), dtype = np.uint8) #tuple
g.shape #tuple的shape不需要加括号
h = g.astype(np.float)


# In[67]:

l = np.arange(10)
m = np.linspace(0,8,5) #且初始化所有元素值为1 --- float
p = np.array([1,2,3,4],[5,6,7,8])

np.save('p.ipynb',p) #保存到文件
q = np.load('p.py') #从文件中读取


# In[74]:

a = np.arange(24).reshape((2,3,4))
print a
b = a[1][1][1] # find one elements
c = a[:,2,:]  # subtract rows

print a[...,1] #竖着取


# In[77]:

f = a[:,1:,1:-1] #从1th取到-1. 最后一个读不到


# In[92]:

g = np.split(np.arange(9),3) # split 3
print g
h = np.split(np.arange(9),[1,-4])  #第一个group取一个，最后一个取4个 ，其余留在中间
print h


# In[97]:

L0 = np.repeat(1,6).reshape((2,3))
L1 = np.arange(6,12).reshape((2,3))
print L0


# In[118]:

# 分大小写的
'''
vstack是指沿着纵轴拼接两个array，vertical
hstack是指沿着横轴拼接两个array，horizontal
更广义的拼接用concatenate实现，horizontal后的两句依次等效于vstack和hstack
stack不是拼接而是在输入array的基础上增加一个新的维度
'''

m = np.vstack((L0,L1)) 
q = np.concatenate((L0,L1)) # == m
#print m


n_horizontal = np.hstack((L0,L1))
r = np.concatenate((L0,L1), axis = 1) # == n_horizontal
#print r

s = np.stack((L0,L1)) #把L1当成一个List1, 把L2当成一个list2，两个合并到一起
print s




# In[140]:

'''

[[[ 1  1  1]
  [ 1  1  1]]

 [[ 6  7  8]
  [ 9 10 11]]]
  
  
---->按指定轴进行转置

array([[[ 0,  3],
        [ 6,  9]],

       [[ 1,  4],
        [ 7, 10]],

       [[ 2,  5],
        [ 8, 11]]])
'''

t = s.transpose((2,0,1)) #竖着反过来

# a = 4*3 --> u = 3*4
a = np.arange(12).reshape(4,3)
u = a.transpose()  # A -> A^T
print u

v = np.rot90(u,3) # 向右转90°
vv =np.rot90(v,3) # 向右转90°
#print vv  #一共转180° 底翻到上面 左翻到右

'''
    [[ 0  3  6  9]
     [ 1  4  7 10]
     [ 2  5  8 11]]    
---->
    [[ 9  6  3  0]
     [10  7  4  1]
     [11  8  5  2]]
'''


w = np.fliplr(u) #左右翻转
print w 


# In[178]:

'''  MATH   '''
a = np.abs(-1)
b = np.sin(np.pi/2)
c = np.arctanh(0.462118)
d = np.exp(3)
f = np.power(2,3) #2^3
h = np.sqrt(25)
sum_list = np.sum([1,2,3,4,5])
mean_list = np.mean([4,5,6,7])
##Variance
p = np.std([1,2,3,1,2,3,1,2,3])

''''Outlier -- Percentile'''

a = np.array([1,2,3,4,5])

percentile_IQ1 = np.percentile(a,25)
percentile_IQ3 = np.percentile(a,75)
IQR = percentile_IQ3- percentile_IQ1

#(IQ1-1.5IQR, IQ3+1.5IQR)

Outlier_Left = percentile_IQ1-1.5*IQR
Outlier_Right = percentile_IQ3+1.5*IQR

print Outlier_Left, Outlier_Right


# In[215]:


##Matrics
dot_time = np.dot([1,2],[3,4]) ## 点积，1*3+2*4=11


a = np.array([
    [1, 2, 3],
    [4, 5, 6]
])

b = np.array([
    [1, 2, 3],
    [1, 2, 3]
])

'''
维度一样的array，对位计算
array([[2, 4, 6],
       [5, 7, 9]])
'''

'''
维度一样的array，对位计算
array([[2, 4, 6],
       [5, 7, 9]])
'''
a + b
a -b 
a*b
a/b

# 变为浮点      ## numpy.ndarray' --- a / a_float都是这个type
a_float = a.astype(np.float)
b_float = b.astype(np.float)
print a_float/b_float

## a**2   a*a 平方
a** 2
a ** b 

## each elements - 1
a-1



# In[ ]:



