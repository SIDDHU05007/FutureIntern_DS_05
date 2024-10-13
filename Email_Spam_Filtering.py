#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[3]:


import pandas as pd
import zipfile
import urllib.request
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# In[13]:


get_ipython().system(' pip install wordcloud')


# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import string


# In[15]:


# Load the dataset
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])


# In[16]:


# 1. Basic statistics and distribution
print("Basic Info:\n", df.info())
print("\nFirst 5 entries:\n", df.head())
print("\nDataset statistics:\n", df.describe())


# In[17]:


# Label distribution (Spam vs. Ham)
sns.countplot(x='label', data=df)
plt.title('Spam vs Ham Distribution')
plt.show()


# In[18]:


# 2. Message length analysis
# Add a new column for message length
df['message_length'] = df['message'].apply(len)


# In[20]:


# Plot message length distribution
plt.figure(figsize=(8, 6))
sns.histplot(df[df['label'] == 'ham']['message_length'], bins=50, label='Ham', color='blue', kde=True)
sns.histplot(df[df['label'] == 'spam']['message_length'], bins=50, label='Spam', color='red', kde=True)
plt.legend()
plt.title('Message Length Distribution (Spam vs Ham)')
plt.show()


# In[21]:


# 3. Word cloud for Spam and Ham messages
def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])
    text = text.lower()
    return text


# In[22]:


# Apply preprocessing
df['processed_message'] = df['message'].apply(preprocess_text)


# In[23]:


# Separate spam and ham messages
spam_words = ' '.join(df[df['label'] == 'spam']['processed_message'])
ham_words = ' '.join(df[df['label'] == 'ham']['processed_message'])


# In[24]:


# Generate word clouds
spam_wc = WordCloud(width=600, height=400, background_color='black').generate(spam_words)
ham_wc = WordCloud(width=600, height=400, background_color='white').generate(ham_words)


# In[25]:


# Display word cloud for Spam messages
plt.figure(figsize=(10, 6))
plt.imshow(spam_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Spam Messages')
plt.show()


# In[26]:


# Display word cloud for Ham messages
plt.figure(figsize=(10, 6))
plt.imshow(ham_wc, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud for Ham Messages')
plt.show()


# In[27]:


# 4. Average message length (Spam vs Ham)
avg_length_spam = df[df['label'] == 'spam']['message_length'].mean()
avg_length_ham = df[df['label'] == 'ham']['message_length'].mean()
print(f"Average message length for Spam: {avg_length_spam:.2f}")
print(f"Average message length for Ham: {avg_length_ham:.2f}")


# In[28]:


# 5. Top words in spam and ham messages (after basic preprocessing)
from collections import Counter


# In[29]:


# Tokenize the processed messages
df['tokens'] = df['processed_message'].apply(lambda x: x.split())


# In[30]:


# Get top words in spam messages
spam_tokens = [token for tokens in df[df['label'] == 'spam']['tokens'] for token in tokens]
top_spam_words = Counter(spam_tokens).most_common(10)


# In[31]:


# Get top words in ham messages
ham_tokens = [token for tokens in df[df['label'] == 'ham']['tokens'] for token in tokens]
top_ham_words = Counter(ham_tokens).most_common(10)


# In[32]:


print("\nTop 10 words in Spam messages:", top_spam_words)
print("Top 10 words in Ham messages:", top_ham_words)


# In[4]:


# Download the ZIP file and extract it
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
zip_file = 'smsspamcollection.zip'
urllib.request.urlretrieve(url, zip_file)

# Extract the ZIP file
with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall()

# Load the dataset from the extracted file
df = pd.read_csv('SMSSpamCollection', sep='\t', header=None, names=['label', 'message'])


# In[5]:


# Preprocessing
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Convert labels to binary
X = df['message']
y = df['label']


# In[6]:


# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[7]:


# Feature extraction using TF-IDF
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)


# In[8]:


# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


# In[9]:


# Make predictions
y_pred = model.predict(X_test_tfidf)


# In[10]:


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


# In[11]:


# Display results
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')


# In[ ]:




