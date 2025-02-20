#!/usr/bin/env python
# coding: utf-8

# **IMPORTING LIBRARIES**

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# **Step 2: Loading and Exploring the Dataset**

# In[2]:


df = pd.read_csv(r"C:\Users\Josphat\Desktop\DATA ANALYSIS VIDEOS AND DATASETS-DATA THINKERS\HEART DISEASE.csv")


# In[3]:


df.head()


# In[4]:


#finding the shape of the dataset
df.shape
print("Number of Rows:", df.shape[0])
print("Number of Columns:", df.shape[1])


# In[5]:


#checking for null values
df.isnull().sum()


# In[6]:


#finding information about the dataset
df.info()


# In[7]:


#finding columns of the dataset
df.columns


# In[8]:


#finding the descriptive statistics of the dataset
df.describe()


# In[9]:


#checking for duplicated values and Dropping them
data_dup=df.duplicated().any()
print(data_dup)


# In[10]:


df.duplicated().sum()


# In[11]:


df.shape


# In[12]:


#dropping the duplicated values
df.drop_duplicates(inplace = True)


# In[13]:


df.shape


# In[14]:


df.describe()


# In[19]:


#Drawing correlation Matrix
plt.figure(figsize=(17,6))
sns.heatmap(df.corr(),annot = True)


# **Question 1: How many people have Heart disease and How many don't have heart disease in this Dataset**

# In[21]:


df.columns


# In[23]:


df["target"].value_counts()


# In[108]:


sns.countplot(x='target', data=df, color='magenta')
plt.xticks([1,0],['Disease','No Disease'])
plt.show()


# `164 people` have heart disease and `138 people` have no heart disease

# **Question 2: Find count of Male and Female in the Dataset**

# In[27]:


df.columns


# In[29]:


df['sex'].value_counts()


# `206 people` are male and `96 people` are Female

# In[40]:


sns.countplot(x='sex', data=df)
plt.xticks([1,0], ['Male','Female'])
plt.show()


# **Question 3: Find gender distribution According to the target variable**

# In[32]:


df.columns


# In[35]:


sns.countplot(x='sex', hue='target', data=df)
plt.xticks([1,0],['Male','Female'])
plt.legend(labels=['No Disease','Disease'])
plt.show()


# **Question 4: Check Age Distribution in the Dataset**

# In[44]:


df.columns


# In[48]:


sns.distplot(df.age, bins=20,color='purple')
plt.show()


# **Question 5: Check Chest pain type**
# 
#     * Value 0: typical angina
#     * Value 1: atypical angina
#     * Value 2: non-anginal pain
#     * Value 3: asymptomatic

# In[49]:


df.columns


# In[54]:


df.cp.value_counts()


# In[56]:


sns.countplot(x='cp', data=df, color='green')
plt.xticks([0,1,2,3],['typical angina','atypical angina','non-anginal pain','asymptomatic'])
plt.xticks(rotation=75)
plt.show()


# **Question 6: Show chest pain Distribution as per Target variabe**

# In[58]:


df.columns


# In[61]:


sns.countplot(x='cp', hue='target',data=df)
plt.xticks([0,1,2,3],['typical angina','atypical angina','non-anginal pain','asymptomatic'],rotation=75)
plt.legend(labels=['No Disease','Disease'])
plt.show()


# In[68]:


df[['cp','target']].value_counts().sort_values(ascending=False)


# **Question 7: Show Fasting Blood Sugar Distribution according to Target variable.**

# In[69]:


df.columns


# In[70]:


df[['fbs','target']].value_counts().sort_values(ascending=False)


# In[73]:


sns.countplot(x='fbs', hue='target',data=df)
plt.legend(labels=['No Disease','Disease'])
plt.show()


# **Question 8: Check Resting Blood Pressure Distribution**

# In[74]:


df.columns


# In[97]:


df.trestbps.hist(color='indigo')
plt.show()


# **Question 9: compare Resting Blood Pressure as per sex column**

# In[102]:


g=sns.FacetGrid(df, hue='sex',aspect=4)
g.map(sns.kdeplot,'trestbps',shade=True)
plt.legend(labels=['Male','Female'])
plt.show()


# **Question 10: Show distribution of Serum Cholestrol**

# In[84]:


df.columns


# In[104]:


df.chol.hist(color='violet')
plt.show()


# **Question 11: Plot Continuous Variables**

# In[90]:


df.columns


# In[91]:


cate_val= []
cont_val= []

for column in df.columns:
    if df[column].nunique() <= 10:
        cate_val.append(column)
    else:
        cont_val.append(column)


# In[92]:


cate_val


# In[93]:


cont_val


# In[106]:


df.hist(cont_val,figsize=(15,6),color='pink')
plt.tight_layout()
plt.show()


# In[ ]:




