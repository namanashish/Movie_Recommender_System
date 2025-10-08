#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np
import streamlit as st
st.title("🚀 Movie Recommender System")
st.write("This app was created by Naman Ashish.")

def local_css(file_name):
    with open (file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style.css")
# In[11]:


movies=pd.read_csv("tmdb_5000_movies.csv")
credits=pd.read_csv("tmdb_5000_credits.csv")


# In[12]:


movies=movies.merge(credits,on='title')


# In[13]:





# In[15]:


movies=movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[16]:





# In[20]:


movies.dropna(inplace=True)


# In[22]:





# In[23]:




# In[27]:


def convert(obj):
    l=[]
    for i in ast.literal_eval(obj) :
        l.append(i['name'])
    return l


# In[26]:


import ast



# In[30]:


movies['genres']=movies['genres'].apply(convert)


# In[31]:


# In[32]:


movies['keywords']=movies['keywords'].apply(convert)


# In[33]:




# In[35]:


def convert3(obj):
    l=[]
    counter=0
    for i in ast.literal_eval(obj) :
        if counter!=3:
            l.append(i['name'])
            counter+=1
        else:
            break
    return l


# In[36]:


movies['cast']=movies['cast'].apply(convert3)


# In[37]:




# In[44]:


def crewz(obj):
    l=[]
    for i in ast.literal_eval (obj):
        if i['job'] == 'Director' :
        
            l.append(i['name'])
            break

    return l


# In[45]:


movies['crew']=movies['crew'].apply(crewz)


# In[46]:




# In[48]:


movies['overview']=movies['overview'].apply(lambda x:x.split())


# In[49]:



# In[54]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace (" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace (" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace (" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace (" ","") for i in x])


# In[55]:




# In[59]:


movies['tags']=movies['genres']+movies['overview']+movies['keywords']+movies['cast']+movies['crew']


# In[81]:


import nltk


# In[86]:


from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()


# In[99]:


def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[88]:


new_df=movies[['movie_id','title','tags']]

# In[89]:





# In[90]:


new_df['tags']=new_df['tags'].apply(lambda x : " ".join(x)) 


# In[91]:


new_df['tags']=new_df['tags'].apply(lambda x:x.lower())


# In[92]:



# In[93]:


from sklearn.feature_extraction.text import CountVectorizer


# In[94]:


cv=CountVectorizer(max_features=5000,stop_words='english')


# In[95]:


vectors=cv.fit_transform(new_df['tags']).toarray()


# In[96]:


# In[104]:


cv.get_feature_names_out() 


# In[98]:


ps.stem('loved')



# In[103]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[105]:


from sklearn.metrics.pairwise import cosine_similarity


# In[108]:


similarity=cosine_similarity(vectors)


# In[119]:


def recommend(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=similarity[movie_index]
    movie_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

    for i in movie_list:
        st.write(new_df.iloc[i[0]].title)


# In[124]:
movs=new_df["title"]
inp=st.text_input("Enter the Movie: ")
select=None
if inp:
    filtered_options=[m for m in movs if inp.lower() in m.lower()]
    if filtered_options:
        select=st.selectbox("Select Movie",["--Select a Movie--"]+filtered_options)
        if select!="Select a Movie" :
            recommend(select)
    else:
        st.write("No movie matches your keyword")






# In[ ]:




