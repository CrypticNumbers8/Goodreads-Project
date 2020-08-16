#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np #Working with arrays
import pandas as pd #DataFrames
import os 
import seaborn as sns #Plotting and Data Visualization
import isbnlib #Data for ISBN strings
from newspaper import Article #For Extracting articles
import matplotlib.pyplot as plt #Module in matplotlib
plt.style.use('ggplot')
from tqdm import tqdm #Progressbars
from progressbar import ProgressBar
import re #Regular Expressions
from scipy.cluster.vq import kmeans,vq #K-means clustering and vector quantization (scipy.cluster.vq)
from pylab import plot,show #Bulk import matplotlib and numpy
from matplotlib.lines import Line2D 
import matplotlib.colors as mcolors
import goodreads_api_client as gr #A lightweight wrapper around the Goodreads API
from sklearn.cluster import KMeans #Kmeans clustering
from sklearn import neighbors #neigbor based learning methods
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler #For feature scaling

import warnings
warnings.filterwarnings("ignore")


# In[ ]:


d1 = pd.read_csv('C:/Users/arvin/OneDrive/Desktop/Completed Projects/Goodreads/books.csv', error_bad_lines = False)
#Drop Bad lines

#print(d1.tail(5))

d1.index = d1['bookID']
print(d1.shape) #Finding number of rows and columns

d1.replace(to_replace = 'J.K. Rowling/Mary GrandPré', value = 'J.K. Rowling', inplace=True)#Removing JK Rowling's co-author name
d1.head()


# In[ ]:


#Exploratory Data Analysis

#Find language counts
sns.set_context('paper')
plt.figure(figsize=(15,10))
ax = d1.groupby('language_code')['title'].count().plot.bar()
plt.title('Language code of books frequency')
plt.xticks(rotation = 45, fontsize = 15)

#Add details inside the graph
for i in ax.patches:
    ax.annotate(str(i.get_height()), (i.get_x()-0.3, i.get_height()+100))

#Conclusion: books in eng, eng_US, eng_GB language are read by almost all the reviwers


# In[ ]:


#Top Books with many publishers
sns.set_context('poster')
plt.figure(figsize = (20,15))
books = d1['title'].value_counts()[:25]#Books having multiple Publishers
#print(books)
rating = d1.average_rating[:25]
#print(rating)
sns.barplot(x=books,y=books.index, palette = 'rocket')
plt.title("Highest frequency Books")
plt.xlabel("High Freq Books")
plt.ylabel("Title of books")
plt.show()

#Conclusion: Dostoyevsky's Brothers Karamazov and The Illiad are published by multiple publishers


# In[ ]:


#TOP 10 Most rated Books
mrated = d1.sort_values('ratings_count',ascending = False).head(10).set_index('title')
plt.figure(figsize = (15,10))
sns.barplot(mrated['ratings_count'], mrated.index, palette = 'deep')

#Conclusion: Twilight #1 has exceptionally high no. of reviewers than other books


# In[ ]:


#Authors with most books
sns.set_context('talk')
big_authors = d1.groupby('authors')['title'].count().reset_index().sort_values('title', ascending = False).head(10).set_index('authors')
plt.figure(figsize = (15,10))
ax1 = sns.barplot(big_authors['title'],big_authors.index,palette = 'rocket')
ax1.set_title("Top 10 Authors with most number of books")
ax1.set_xlabel("total no of books")

#Conclusion: King,Wodehouse and Takahashi have written the most number of books


# In[ ]:


d1['authors'].value_counts().head(10)


# In[ ]:


#Most Highly rated Authors

high_authors = d1[d1['average_rating']>= 4.4]
high_authors = high_authors.groupby('authors')['title'].count().reset_index().sort_values('title', ascending = False).head(10).set_index('authors')
plt.figure(figsize = (15,10))
ax2 = sns.barplot(high_authors['title'], high_authors.index, palette = 'icefire_r')
ax2.set_xlabel("No. of BOOKS")
ax2.set_ylabel("Authors")

#Conclusion: Rowling, Watanabe are the 2 most highly rated authors among the reviewers


# In[ ]:


d1.average_rating.isnull().value_counts()
d1.dropna(0, inplace=True) #Removing any null values

plt.figure(figsize = (15,15))
ratings = d1.average_rating.astype('float')
sns.distplot(ratings, bins = 20)

#Conclusion: Distplot shows that most of the ratings are nearby 4


# In[ ]:


#Rating segregation of books
def segregation(data):
    val = []
    for i in data.average_rating:
        if i>=0 and i<=1:
            val.append("From 0 to 1")
        elif i>1 and i<=2:
            val.append("From 1 to 2")
        elif i>2 and i<=3:
            val.append("From 2 to 3")
        elif i>3 and i<=4:
            val.append("From 3 to 4")
        elif i>4 and i<=5:
            val.append("From 4 to 5")
        else:
            val.append("NaN")
    print(len(val))
    return val
    
d1['Ratings_Dist'] = segregation(d1)
d1.head()


# In[ ]:


#Pie Chart based on ratings
pies = d1['Ratings_Dist'].value_counts().reset_index()
#print(pies)
labels = pies['index']
colors = ['blue','green','red','black','yellow']
percent = 100.*pies['Ratings_Dist']/pies['Ratings_Dist'].sum()
#print(labels)
#print(percent)
fig, ax3 = plt.subplots()
ax3.pie(pies['Ratings_Dist'],colors = colors, pctdistance = 0.75, startangle = 90, explode = (0.05, 0.05,0.05, 0.05, 0.05))
centre = plt.Circle((0,0), 0.5, fc = 'white')
fig1 = plt.gcf()
fig1.gca().add_artist(centre)
plt.axis('equal')
plt.tight_layout()
plt.legend(labels, loc = 'upper left', shadow = True, prop = {'size':9},fontsize = 'x-small')

#Conclusion: Ratings from 0 to 3 are extremely rare because if people rate at least 3 even if they end up finishing a bad book


# In[ ]:


#Relation between ratings and review counts
plt.figure(figsize = (15,10))
sns.set_context('paper')
ax4 = sns.jointplot(x="average_rating", y = 'text_reviews_count', kind = 'scatter', data = d1[['text_reviews_count', 'average_rating']])
ax4.set_axis_labels("Average Rating", "Text Review Count")
plt.show()

#Conclusion: Most amounts of reviews(~5000) are between 3 and 4


# In[ ]:


trial = d1[~(d1['text_reviews_count']>5000)]
plt.figure(figsize = (15,10))
d1.dropna(0, inplace = True)
sns.set_context('paper')
ax5 = sns.jointplot(x = 'average_rating', y = 'text_reviews_count', kind = 'scatter', data = trial, color = 'blue')
ax5.set_axis_labels("Average Rating", "text review count")
plt.show()

#Conclusion: Many books that have rating between 3-4 are having 0-2000 reviews


# In[ ]:


#Relation between number of pages and ratings
sns.set_context('paper')
ax6 = sns.jointplot(x= "average_rating", y = "  num_pages", data = d1, color = 'green')
ax6.set_axis_labels("Average Rating", "Number of Pages")

#Counclusion: The page count below 1000 has maximum no of ratings as people don't like to read thick books


# In[ ]:


trial1 = d1[~(d1['  num_pages']>1000)]
ax7 = sns.jointplot(x = 'average_rating', y = '  num_pages', data = trial1, color='maroon')
ax7.set_axis_labels("Average Rating", "Number of pages")

#Most ratings are given for page range of 150-400, which shows that thicker books are read rarer among the general public


# In[ ]:


#Relation between ratings and rating counts
sns.set_context('paper')
ax8 = sns.jointplot(x = 'average_rating', y ='ratings_count', data = d1, color = 'black')
ax8.set_axis_labels("average ratings", "no of ratings")

#Conclusion: For a better outlook, let's consider the count around 2000000 ratings


# In[ ]:


trial3 = d1[~(d1['ratings_count']>1000000)]
ax9 = sns.jointplot(x='average_rating', y = 'ratings_count', data = trial3, color = 'blueviolet')
ax9.set_axis_labels("average ratings", "no of ratings")

#Conclusion: As the no of ratings increase, rating for book asymptotes towards rating=4


# In[ ]:


#Books with highest reviews
most_text = d1.sort_values('text_reviews_count', ascending = False).head(10).set_index('title')
plt.figure(figsize = (15,10))
sns.set_context('poster')
ax10 = sns.barplot(most_text['text_reviews_count'], most_text.index, palette = 'rocket')
plt.show()

#Conclusion: Twilight and The Book Thief has the most no of reviews


# In[ ]:


#Find a relationship between the rating count and avg rating

trial4 = d1[['average_rating', 'ratings_count']]
data = np.asarray([np.asarray(trial4['average_rating']), np.asarray(trial4['ratings_count'])]).T
print(data)


# In[ ]:


#Using elbow curve method to find no. of clusters for data

x = data
distortions = []
for k in range(2,30):
    k_means = KMeans(n_clusters = k)
    k_means.fit(x)
    distortions.append(k_means.inertia_)
    
fig = plt.figure(figsize = (15,10))
plt.plot(range(2,30), distortions, 'bx-')
plt.title("Elbow Curve")

#Elbow lies around value K=5


# In[ ]:


#Computing K means with K = 5

centroids,_ = kmeans(data,5)

#assigning each sample to a cluster
#Vector Quantisation:
idx,_ = vq(data,centroids)


# In[ ]:


#Computing K means with K = 5, thus, taking it as 5 clusters
centroids, _ = kmeans(data, 5)

#assigning each sample to a cluster
#Vector Quantisation:

idx, _ = vq(data, centroids)
# some plotting using numpy's logical indexing
sns.set_context('paper')
plt.figure(figsize=(15,10))
plt.plot(data[idx==0,0],data[idx==0,1],'or',#red circles
     data[idx==1,0],data[idx==1,1],'ob',#blue circles
     data[idx==2,0],data[idx==2,1],'oy', #yellow circles
     data[idx==3,0],data[idx==3,1],'om', #magenta circles
     data[idx==4,0],data[idx==4,1],'ok',#black circles
        )
plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8)

circle1 = Line2D(range(1), range(1), color = 'red', linewidth = 0, marker= 'o', markerfacecolor='red')
circle2 = Line2D(range(1), range(1), color = 'blue', linewidth = 0,marker= 'o', markerfacecolor='blue')
circle3 = Line2D(range(1), range(1), color = 'yellow',linewidth= 0,  marker= 'o', markerfacecolor='yellow')
circle4 = Line2D(range(1), range(1), color = 'magenta', linewidth= 0,marker= 'o', markerfacecolor='magenta')
circle5 = Line2D(range(1), range(1), color = 'black', linewidth = 0,marker= 'o', markerfacecolor='black')

plt.legend((circle1, circle2, circle3, circle4, circle5)
           , ('Cluster 1','Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'), numpoints = 1, loc = 0, )


plt.show()


# In[ ]:


#Finding outliers and removing them
trial4.idxmax()


# In[ ]:


trial4.drop(41865, inplace = True)


# In[ ]:


data = np.asarray([np.asarray(trial['average_rating']), np.asarray(trial['ratings_count'])]).T
#Computing K means with K = 8, thus, taking it as 8 clusters
centroids, _ = kmeans(data, 5)

#assigning each sample to a cluster
#Vector Quantisation:

idx, _ = vq(data, centroids)
# some plotting using numpy's logical indexing
sns.set_context('paper')
plt.figure(figsize=(15,10))
plt.plot(data[idx==0,0],data[idx==0,1],'or',#red circles
     data[idx==1,0],data[idx==1,1],'ob',#blue circles
     data[idx==2,0],data[idx==2,1],'oy', #yellow circles
     data[idx==3,0],data[idx==3,1],'om', #magenta circles
     data[idx==4,0],data[idx==4,1],'ok',#black circles
    
     
        
        
        
        
        )
plt.plot(centroids[:,0],centroids[:,1],'sg',markersize=8, )




circle1 = Line2D(range(1), range(1), color = 'red', linewidth = 0, marker= 'o', markerfacecolor='red')
circle2 = Line2D(range(1), range(1), color = 'blue', linewidth = 0,marker= 'o', markerfacecolor='blue')
circle3 = Line2D(range(1), range(1), color = 'yellow',linewidth=0,  marker= 'o', markerfacecolor='yellow')
circle4 = Line2D(range(1), range(1), color = 'magenta', linewidth=0,marker= 'o', markerfacecolor='magenta')
circle5 = Line2D(range(1), range(1), color = 'black', linewidth = 0,marker= 'o', markerfacecolor='black')

plt.legend((circle1, circle2, circle3, circle4, circle5)
           , ('Cluster 1','Cluster 2', 'Cluster 3', 'Cluster 4', 'Cluster 5'), numpoints = 1, loc = 0, )


plt.show()

#Conclusion: The green squares are the centroids for the given clusters. 
#As the rating count seems to decrease, the average rating seems to become sparser, with higher volatility and less accuracy.


# In[ ]:


#Building the recommendation system
books_features = pd.concat([d1['Ratings_Dist'].str.get_dummies(sep=","), d1['average_rating'], d1['ratings_count']], axis=1)


# In[ ]:


books_features.head()


# In[ ]:


min_max_scaler = MinMaxScaler()
books_features = min_max_scaler.fit_transform(books_features)
np.round(books_features, 2)


# In[ ]:


model = neighbors.NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
model.fit(books_features)
distance, indices = model.kneighbors(books_features)


# In[ ]:


def get_index_from_name(name):
    return d1[d1["title"]==name].index.tolist()[0]

all_books_names = list(d1.title.values)

def get_id_from_partial_name(partial):
    for name in all_books_names:
        if partial in name:
            print(name,all_books_names.index(name))
            
def print_similar_books(query=None,id=None):
    if id:
        for id in indices[id][1:]:
            print(d1.iloc[id]["title"])
    if query:
        found_id = get_index_from_name(query)
        for id in indices[found_id][1:]:
            print(d1.iloc[id]["title"])


# In[ ]:


print_similar_books("The Catcher in the Rye")


# In[ ]:


print_similar_books("The Brothers Karamazov")


# In[ ]:


get_id_from_partial_name("Harry Potter and the ")


# In[ ]:


print_similar_books(id = 1) #ID for the Book 5


# In[ ]:


#_______________

