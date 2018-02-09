
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')


# In[10]:


from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


# In[35]:


get_ipython().system(u' conda install pyflux')


# In[40]:


tsla_org = pd.read_csv("tsla.csv", sep=",")


# In[44]:


tsla = pd.DataFrame(np.diff(np.log(tsla_org['Adj Close']))[500:len(tsla_org['Adj Close'])])
tsla.index = pd.to_datetime(tsla_org['Date'].values[1+500:len(tsla_org)])
tsla.columns = ['AdjClose']
plt.figure(figsize=(15,5))
plt.plot(tsla)
plt.ylabel('Tesla')
plt.title('Tesla');


# In[45]:


model = pf.GPNARX(tsla,ar=4,kernel_type='OU')
x = model.fit()
x.summary()

