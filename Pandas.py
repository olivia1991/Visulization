
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[5]:

s = pd.Series([1,3,5,np.nan,6,8])


# In[3]:

#创建一个data.fram

dates = pd.date_range('2013-06-01',periods = 6)
df = pd.DataFrame(np.random.randn(6,4), index = dates, columns = list('ABCD'))


# In[33]:

df


# In[46]:

df2 = pd.DataFrame(
                {'A': 1. ,
                 'B': pd.Timestamp('20130102'),
                 'C': pd.Series(1, index=list(range(4)),dtype = 'float32'),
                 'D': np.array([3]*4, dtype = 'int32'),
                 'E': pd.Categorical(['test','train','test','train']),
                 'F':'foo'
                }
                )


# In[47]:

df2


# In[51]:

df2.dtypes   #types(df2) --> check 对df2整体的datafram. --> data.fram 的是dtypes


# In[67]:

# Top 10 Lines // End 10 Line
df.head(1)
df.tail(2)
df.columns  #  R.names()

df.index   ## 行号， 其实就是dataframe的序列
df.values


# In[70]:

df.describe()   #R.summary


# In[71]:

df.T


# In[74]:

##Sort
df.sort_index(axis = 0, ascending = False) #按row排列
df.sort_index(axis =1, ascending = False) # 按column索引


# In[81]:

df.sort_index(by = ['A','B'], ascending = [False,True])


# In[83]:

df.sort(columns = 'B')


# In[85]:

#cutting
df['A']
df[0:3]


# In[99]:

#loc是根据dataframe的具体标签选取列，而iloc是根据标签所在的位置，从0开始计数。
#需要注意的是，如果是df = df.loc[0:2, ['A', 'C']]或者df = df.loc[0:2, ['A', 'C']]，切片之后依旧是一个dataframe。
#是不能进行加减乘除等操作的，比如dataframe的一列是数学成绩(shuxue)，另一列为语文成绩(yuwen)，现在需要求两门课程的总和
#如果你想要选取某一行的数据，可以使用df.loc[[i]]或者df.iloc[[i]]。

df.loc[dates[0]] ##取了时间日期为2013-06-01的那一行
df.loc[: ,['A','B']]  ## colums/row 要框起来！！！！
df.loc['20130602':'20130604',['A','B']]

#获取一个量.两个相同
df.loc[dates[0],'A']
df.at[dates[0],'A']


# In[108]:

#loc更加具体标签选择  
#iloc 根据数字选择
df.iloc[3]   # -- row
df.iloc[:,[3]] #-- columns

df.iloc[3:5, 0:2]
df.iloc[[1,2,4],[0,2]]
df.iloc[1,1] #取特定的值
df.iat[1,1]  #取特定值 同上


# In[112]:

'''    Boolean   '''

df[df.A > 0]  #取A列大于0的


# In[189]:

B = [['A','B','C'],[1,2,3]]
d = np.array(B).reshape(2,3)
df_fake = pd.DataFrame(d, columns = list('ABC'))

#df = pd.DataFrame(np.array(my_list).reshape(3,3), columns = list("abc"))


# In[183]:

##Filter!!!!!!
dff = [{'A': 1,'B': 2,'C': 'iPHONE'},{'A':4 ,'B':3,'C':'Google'},{'A':5,'B':3,'C':'Google'}]

[i['A'] for i in dff if i['C'] == 'Google'][0]


# In[195]:

""" 用 isin() 过滤"""
df2 = df.copy()
df2['E'] = ['one','one','two','three','four','three']
df2


# In[197]:

df2[df2['E'].isin(['one','two'])]


# In[209]:

""" 加一个新列  """
s1 = pd.Series([1,2,3,4,5,6], index = pd.date_range('20130601', periods = 6)) 
##  一定要给出这个index。 
##  index要写出这个表的行号。要不然就是NA
df2['F'] = s1


# In[208]:

df2


# In[226]:

# 通过标签设新的值
df2.at[dates[0],'A'] = 0  #和python不一样 要加 ”at“

#df[df[2,2]] = 99  --> 这样是加了一个新的column(2,2)

#通过位置【具体i,j】设计值
df2.iat[0,1] = 9


# In[225]:

df2


# In[228]:

#通过一个numpy数组设置一组新值：

df2.loc[:,'D'] = np.array([5]*len(df2))


# In[239]:

#通过where设值 boolean

df3 = df.copy()
df[df[2,2]] = 99

df3[df3>0] = 5


# In[236]:

df3


# # 缺失值处理

# In[40]:

df1 = df.reindex(index = dates[0:4])


# In[41]:

df1.at[dates[0]:dates[1],'E'] = 1
df.loc[dates[0]:dates[1],'E'] =1  #等价


# In[62]:

df1


# In[61]:

df1.dropna(axis = 0)  #drop 横向有NA的值, 不会对原始值进行改动
df1.dropna(how = 'any')
df1.dropna(axis = 1)  #drop 纵向有NA的值


# In[64]:

df.fillna(value = 5)


# In[65]:

pd.isnull(df1)


# # 相关操作

# In[83]:

df.iloc[1,:]
df.ix[[1,2]]
df.ix[:, 1].mean()
df.mean()  #取每一列的均值
df.mean(1) #每一行值


# In[111]:


s = pd.Series([1,3,5,np.nan,6,8], index = dates).shift(3) #向下移动两位
print s
print df
print df.sub(s,axis = 'index') ##按index,实际就是没有列，都要与s对减
#df.sub(row, axis='columns') 例如row是第二行，则axis = 'column'意思是
#每一行 都与此行对减


# In[117]:

##Apply##
df.apply(np.cumsum) ##列 累计求和 第二列等于row1+row2
df.apply(lambda x: x.max() - x.min()) #每一列




# In[119]:

a = np.array([[1,2,3], [4,5,6]])
a
np.cumsum(a)
np.cumsum(a, dtype=float)
np.cumsum(a,axis=0)      # sum over rows for each of the 3 columns
np.cumsum(a,axis=1)      # sum over columns for each of the 2 rows


# In[122]:

##Histogram
s = pd.Series(np.random.randint(0,7,size = 10))
s.value_counts()


# In[125]:

s = pd.Series(['A','B','C','Aaba','Baca',np.nan,'CABA','dog','cat'])
s.str.lower()


# # Merge

# In[130]:

df = pd.DataFrame(np.random.randn(10,4))
df


# In[137]:

pieces = [df[:3],df[3:7],df[7:]]


# In[139]:

pd.concat(pieces)


# In[140]:

## Join ##
left = pd.DataFrame({'key':['foo','foo'], 'lval':[1,2]})
right = pd.DataFrame({'key':['foo','foo'],'rval':[4,5]})


# In[143]:

pd.merge(left, right, on = 'key')


# In[153]:

## Append ##
df = pd.DataFrame(np.random.randn(8,4), columns = ['A','B','C','D'])
df
s = df.iloc[3]


# In[158]:

#将一列连在另一列上面
df.append(s, ignore_index = False)
df.append(s, ignore_index = True)


# # 分组
# # 对于”group by”操作，我们通常是指以下一个或多个操作步骤：
# 
# # l  （Splitting）按照一些规则将数据分为不同的组；
# 
# # l  （Applying）对于每组数据分别执行一个函数；
# 
# # l  （Combining）将结果组合到一个数据结构中；

# In[161]:

df = pd.DataFrame({'A': ['foo','bar','foo','bar','foo','bar','foo','foo'],
                   'B':  ['one','one','two','three','two','two','one','three'],
                   'C': np.random.randn(8),
                   'D': np.random.randn(8)})


# In[168]:

df
df.groupby('A').sum()
df.groupby(['A','B']).sum()


# # Reshaping

# In[169]:

tuples = list(zip(*[['bar','bar','baz','baz','foo','foo','qux','qux'],
                    ['one','two','one','two','one','two','one','two']]))


# In[174]:

index = pd.MultiIndex.from_tuples(tuples, names=['first','second'])
df = pd.DataFrame(np.random.randn(8,2), index = index, columns = ['A','B'])


# In[182]:

df2 = df[:4]
#df3 = df.iloc[1,:]


# In[187]:

stacked = df2.stack()  #---> series
#转到左边，作为行变量


# In[190]:

stacked.unstack()  # ---> data.frame


# In[197]:

stacked.unstack()
# unstack 转到右边 作为列变量
#df.unstack('number').stack('number') 
#更改了multiindex的顺序。先把'number'作为列的index，再拿下来作为行的index，就更改了在index中出现的顺序


# # Pivotal Table

# In[203]:

df = pd.DataFrame({'Number': ['one','one','two','three']*3,
                   'Char' : ['A','B','C']*4,
                   'C' : ['foo','foo','foo','bar','bar','bar']*2,
                   'D' : np.random.randn(12),
                   'E' : np.random.randn(12)})


# In[204]:

pd.pivot_table(df, values = 'D', index = ['Number','Char'], columns = ['C'])


# # 时间序列

# In[216]:

rng = pd.date_range('1/1/2012',periods =100,freq = 'S') 
## 3D 就是代表每3天


# In[218]:

ts = pd.Series(np.random.randint(0,500,len(rng)), index = rng)


# In[223]:

ts.resample('5Min').sum()   #按秒采样的数据转换为按5分钟为单位进行采样的数据


# In[225]:

rng = pd.date_range('3/6/2012 00:00', periods = 5, freq = 'D')


# In[227]:

ts = pd.Series(np.random.randn(len(rng)), rng)


# In[232]:

ts_utc = ts.tz_localize('UTC')


# In[233]:

ts_utc


# In[234]:

ts_utc.tz_convert('US/Eastern') #时区转换：


# In[ ]:



