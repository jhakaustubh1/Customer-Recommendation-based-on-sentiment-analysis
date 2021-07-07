#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df=pd.read_csv('1429_1.csv')
df.head(5)


# In[4]:



dt1=df[["name","categories","reviews.rating","reviews.text","reviews.username"]]
dt1["reviews.rating"].replace({1.0:0.0},inplace=True)
dt1["reviews.rating"].replace({2.0:0.0},inplace=True)
dt1["reviews.rating"].replace({3.0:0.0},inplace=True)
dt1["reviews.rating"].replace({4.0:1.0},inplace=True)
dt1["reviews.rating"].replace({5.0:1.0},inplace=True)
dt1.head(5)


# In[5]:


dt1.isnull().sum()


# In[6]:


dt=dt1.dropna(axis=0,how='any')
dt.isnull().sum()


# In[7]:


dt['reviews.username'].value_counts()


# In[8]:


#removing punctuation,numbers and special characters
dt['reviews.text']=dt['reviews.text'].str.replace("[^a-zA-Z#]"," ")
dt.head(5)


# In[13]:


import nltk
nltk.download('punkt')
#Removing stop words and tokenizing the text
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
dt['reviews.text']=dt['reviews.text'].astype(str)
s=dt['reviews.text']
stop_words=stopwords.words('english')
filtered_reviews=[]
for text in s:
    word_tokens=word_tokenize(text)
    word_tokens
    filtered_sentence=[w for w in word_tokens if not w in stop_words]
    filtered_reviews.append(filtered_sentence)
filtered_reviews


# In[14]:


from nltk import PorterStemmer
ps=PorterStemmer()
stemmed_reviews=[]
for text in filtered_reviews:
    L=[]
    for w in text:
        L.append(ps.stem(w))
    stemmed_reviews.append(L)
stemmed_reviews


# In[15]:


clean=[]
for text in stemmed_reviews:
    st=""
    for w in text:
        st=st+w+" "
    clean.append(st)
#stemmed_reviews
dt['Cleaned reviews']=[rev for rev in clean]


# In[16]:


dt.head(5)


# In[ ]:





# In[17]:


from sklearn.feature_extraction.text import CountVectorizer
bvect=CountVectorizer(max_df=0.9,min_df=2,stop_words='english',ngram_range=(1,2))
bag_of_words=bvect.fit_transform(dt['Cleaned reviews'])
#df_bword=pd.DataFrame(bag_of_words)
#df_bword


# In[18]:


train_bag=bag_of_words[:27867]
Y=dt.iloc[0:27867,2]
from sklearn.model_selection import train_test_split
x_train_bag,x_valid_bag,y_train_bag,y_valid_bag=train_test_split(train_bag,Y,test_size=0.3,random_state=2)


# In[19]:


from sklearn.tree import DecisionTreeClassifier
dc=DecisionTreeClassifier(criterion='entropy',random_state=1).fit(x_train_bag,y_train_bag)
dc_pred=dc.predict(x_valid_bag)
from sklearn.metrics import f1_score
f1_score(y_valid_bag,dc_pred,average='weighted')


# In[20]:


sent=["Bad product","Good product"]
X_new_counts=bvect.transform(sent)
predicted=dc.predict(X_new_counts)
predicted


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer=TfidfTransformer(use_idf=False).fit(bag_of_words)
X_train_tf=tf_transformer.transform(bag_of_words)


# In[22]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tf, dt['reviews.rating'])


# In[24]:


pred2=clf.predict(tf_transformer.transform(x_valid_bag))
f1_score(y_valid_bag,pred2,average='weighted')


# In[25]:


docs_new=['Bad product','Good product']
X_new_counts=bvect.transform(docs_new)
X_new_tfidf=tf_transformer.transform(X_new_counts)
predicted=clf.predict_proba(X_new_tfidf)*100
predicted


# In[26]:


#Grouped bar chart for comparing probabilities of positive and negative reviews
pro1=[predicted[0][0],predicted[1][0]]
pro2=[predicted[0][1],predicted[1][1]]
ypos=np.arange(2)
plt.bar(ypos,tuple(pro1),width=0.3,label="Negative",color="red")
plt.bar(ypos+0.3,tuple(pro2),width=0.3,label="Positive",color="green")
plt.title("Probability of positive and negative sentiment")
plt.xlabel("Review")
plt.ylabel("Probability")
plt.xticks(ypos+0.3/2,(docs_new[0],docs_new[1]))
plt.legend(labels=["Negative","Positive"])
plt.show()


# In[ ]:





# In[27]:


dts=dt[dt['reviews.rating']==0.0]
dts.head()


# In[28]:


import string


# In[29]:


def clean_string(text):
    text=''.join([word for word in text if word not in string.punctuation])
    text=text.lower()
    text=' '.join([word for word in text.split() if word not in set(stopwords.words("english"))])
    return text


# In[30]:


sentences=dts['reviews.text']
cleaned=list(map(clean_string,sentences))


# In[25]:


sentences


# In[31]:


cleaned


# In[32]:


from sklearn.metrics.pairwise import cosine_similarity
vectorizer=CountVectorizer().fit_transform(cleaned)
vectors=vectorizer.toarray()


# In[33]:


#calculating cosine_similarity between two reviews
def cosine_sim_vectors(vec1,vec2):
    vec1=vec1.reshape(1,-1)
    vec2=vec2.reshape(1,-1)
    return cosine_similarity(vec1,vec2)[0][0]


# In[34]:


csim_max=[]
for i in range(0,len(vectors)):
    csim_max.append(cosine_sim_vectors(vectors[0],vectors[i]))
csim_max


# In[35]:


dts["cos sim"]=csim_max
rstdf=dts.sort_values(by="cos sim",ascending=False)


# In[36]:


rstdf.head(5)
d1=rstdf.iloc[1:6,]


# In[37]:


plt.rcdefaults()
fig,ax=plt.subplots(figsize=(10,10))
y_pos=np.arange(len(d1["reviews.text"]))
error=np.random.rand(len(d1["reviews.text"]))
ax.barh(y_pos,d1["cos sim"],xerr=error,align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(d1["reviews.text"])
ax.invert_yaxis()
ax.set_xlabel("Cosine similarity")
ax.set_title("Similar reviews")
plt.show()


# In[38]:


#recommendation based on sentiment analysis
dt['name'].value_counts()


# In[39]:


len(dt)


# In[ ]:




