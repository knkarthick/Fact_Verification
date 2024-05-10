import os
os.chdir(r"D:\Documents\Fact_Verification")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json


df = pd.read_csv("./data/train.csv")
print("No. of Rumors: ", len(df['id'].unique()))
print("No. of User IDs: ", len(df['username'].unique()))

print("Evidence to No Evidence Proportion: ", df['is_evidence'].value_counts())
print("Evidence to No Evidence Proportion (%): ", df['is_evidence'].value_counts(normalize=True))


ax = df.label.value_counts().plot(kind='barh')
plt.xlabel('Count #')


# Rumor Length
df['rumor_length'] = [len(x.split(" ")) for x in df.rumor]
dfrl = df['rumor_length']
max(df.rumor_length)
fig, axes = plt.subplots()
fig.suptitle("Rumor Length")
plt.hist(df.rumor_length, bins=50, label='Rumor Length')
fig.legend()
fig, axes = plt.subplots()
fig.suptitle("Rumor Length")
sns.distplot(df['rumor_length'], label='Rumor Length')
fig.legend()


# Timeline Length
df['timeline_length'] = [len(x.split(" ")) for x in df.timeline]
dftl = df['timeline_length']
max(df.timeline_length)
fig, axes = plt.subplots()
fig.suptitle("Timeline Length")
plt.hist(df.timeline_length, bins=50, label='Timeline Length')
fig.legend()
fig, axes = plt.subplots()
sns.distplot(df['timeline_length'], label='Timeline Length')
fig.legend()
#%%

'''
Data Cleaning Process: [Help - Internet]
- Decoding: unicode_escape for extra “\” before unicode character, then unidecode
- Apostrophe handled: there are two characters people use for contraction. “’”(apostrophe) and “‘“(single quote). If these two symbols are both used for contraction, it will be difficult to detect and properly map the right expanded form. So any “’”(apostrophe) is changed to “‘“(single quote)
- Contraction check: check if there’s any contracted form, and replace it with its original form
- Parsing: done with Spacy
- Filtering punctuation, white space, numbers, URL using Spacy methods while keeping the text content of hashtag intact
- Removed @mention
- Lemmatize: lemmatized each token using Spacy method ‘.lemma_’. Pronouns are kept as they are since Spacy lemmatizer transforms every pronoun to “-PRON-”
- Special character removal
- Single syllable token removal
- Spell correction: it is a simple spell correction dealing with repeated characters such as “sooooo goooood”. If the same character is repeated more than two times, it shortens the repetition to two. For example “sooooo goooood” will be transformed as “soo good”. This is not a perfect solution since even after correction, in case of “soo”, it is not a correct spelling. But at least it will help to reduce feature space by making “sooo”, “soooo”, “sooooo” to the same word “soo”
'''
contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", 
                   "can't've": "cannot have", "'cause": "because", "could've": "could have", 
                   "couldn't": "could not", "couldn't've": "could not have","didn't": "did not", 
                   "doesn't": "does not", "don't": "do not", "hadn't": "had not", 
                   "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not", 
                   "he'd": "he would", "he'd've": "he would have", "he'll": "he will", 
                   "he'll've": "he will have", "he's": "he is", "how'd": "how did", 
                   "how'd'y": "how do you", "how'll": "how will", "how's": "how is", 
                   "I'd": "I would", "I'd've": "I would have", "I'll": "I will", 
                   "I'll've": "I will have","I'm": "I am", "I've": "I have", 
                   "i'd": "i would", "i'd've": "i would have", "i'll": "i will", 
                   "i'll've": "i will have","i'm": "i am", "i've": "i have", 
                   "isn't": "is not", "it'd": "it would", "it'd've": "it would have", 
                   "it'll": "it will", "it'll've": "it will have","it's": "it is", 
                   "let's": "let us", "ma'am": "madam", "mayn't": "may not", 
                   "might've": "might have","mightn't": "might not","mightn't've": "might not have", 
                   "must've": "must have", "mustn't": "must not", "mustn't've": "must not have", 
                   "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock", 
                   "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                   "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", 
                   "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", 
                   "she's": "she is", "should've": "should have", "shouldn't": "should not", 
                   "shouldn't've": "should not have", "so've": "so have","so's": "so as", 
                   "this's": "this is",
                   "that'd": "that would", "that'd've": "that would have","that's": "that is", 
                   "there'd": "there would", "there'd've": "there would have","there's": "there is", 
                       "here's": "here is",
                   "they'd": "they would", "they'd've": "they would have", "they'll": "they will", 
                   "they'll've": "they will have", "they're": "they are", "they've": "they have", 
                   "to've": "to have", "wasn't": "was not", "we'd": "we would", 
                   "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", 
                   "we're": "we are", "we've": "we have", "weren't": "were not", 
                   "what'll": "what will", "what'll've": "what will have", "what're": "what are", 
                   "what's": "what is", "what've": "what have", "when's": "when is", 
                   "when've": "when have", "where'd": "where did", "where's": "where is", 
                   "where've": "where have", "who'll": "who will", "who'll've": "who will have", 
                   "who's": "who is", "who've": "who have", "why's": "why is", 
                   "why've": "why have", "will've": "will have", "won't": "will not", 
                   "won't've": "will not have", "would've": "would have", "wouldn't": "would not", 
                   "wouldn't've": "would not have", "y'all": "you all", "y'all'd": "you all would",
                   "y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                   "you'd": "you would", "you'd've": "you would have", "you'll": "you will", 
                   "you'll've": "you will have", "you're": "you are", "you've": "you have" } 

import codecs
import unidecode
import re
import spacy
# import spacy.cli
# spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")



def spacy_cleaner(text):
    try:
        decoded = unidecode.unidecode(codecs.decode(text, 'unicode_escape'))
    except:
        decoded = unidecode.unidecode(text)
    apostrophe_handled = re.sub("’", "'", decoded)
    expanded = ' '.join([contraction_mapping[t] if t in contraction_mapping else t for t in apostrophe_handled.split(" ")])
    parsed = nlp(expanded)
    final_tokens = []
    for t in parsed:
        if t.is_punct or t.is_space or t.like_num or t.like_url or str(t).startswith('@'):
            pass
        else:
            if t.lemma_ == '-PRON-':
                final_tokens.append(str(t))
            else:
                sc_removed = re.sub("[^a-zA-Z]", '', str(t.lemma_))
                if len(sc_removed) > 1:
                    final_tokens.append(sc_removed)
    joined = ' '.join(final_tokens)
    spell_corrected = re.sub(r'(.)\1+', r'\1\1', joined)
    return spell_corrected

# CLEANING THE TWEETS
df['rumor_cleaned'] = df['rumor'].apply(spacy_cleaner)
df['timeline_cleaned'] = df['timeline'].apply(spacy_cleaner)
df['rumor_cleaned_length'] = [len(x.split(" ")) for x in df.rumor_cleaned]
dfrcl = df['rumor_cleaned_length']
df['timeline_cleaned_length'] = [len(x.split(" ")) for x in df.timeline_cleaned]
dftcl = df['timeline_cleaned_length']
fig, axes = plt.subplots()
fig.suptitle("Rumor Length (Original Vs Cleaned")
sns.distplot(df['rumor_length'], label='Original Data')
sns.distplot(df['rumor_cleaned_length'], label='Cleaned Data')
fig.legend()

fig, axes = plt.subplots()
fig.suptitle("Timeline Length (Original Vs Cleaned")
sns.distplot(df['timeline_length'], label='Original Data')
sns.distplot(df['timeline_cleaned_length'], label='Cleaned Data')
fig.legend()
#%%
# HANDLING THE IMBALANCED DATASETS
def compute_class_weight(train_y):
    """
    Compute class weight given imbalanced training data
    Usually used in the neural network model to augment the loss function (weighted loss function)
    Favouring/giving more weights to the rare classes.
    """
    import sklearn.utils.class_weight as scikit_class_weight

    class_list = list(set(train_y))
    class_weight_value = scikit_class_weight.compute_class_weight(class_weight ='balanced', classes = class_list, y = train_y)
    class_weight = dict()

    # Initialize all classes in the dictionary with weight 1
    curr_max = int(np.max(class_list))
    for i in range(curr_max):
        class_weight[i] = 1

    # Build the dictionary using the weight obtained the scikit function
    for i in range(len(class_list)):
        class_weight[class_list[i]] = class_weight_value[i]

    return class_weight

weights = compute_class_weight(df['is_evidence'])

import torch
import torch.nn as nn

weight_list = []
for key, weight in weights.items():
    weight_list.append(weight)
weight_tensor = torch.FloatTensor(weight_list)

# with weights
loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)


from imblearn.over_sampling import RandomOverSampler
oversample = RandomOverSampler(sampling_strategy='minority')

df, df['is_evidence'] = oversample.fit_resample(df[['id', 'rumor', 'label', 'username', 'timeline_id',
       'timeline', 'rumor_length', 'timeline_length']], df['is_evidence'])
df['rumor_cleaned_balanced'] = df['rumor'].apply(spacy_cleaner)
df['timeline_cleaned_balanced'] = df['timeline'].apply(spacy_cleaner)
df['rumor_cleaned_length_balanced'] = [len(x.split(" ")) for x in df.rumor_cleaned_balanced]
df['timeline_cleaned_length_balanced'] = [len(x.split(" ")) for x in df.timeline_cleaned_balanced]


fig, axes = plt.subplots(figsize=(12,10))
fig.suptitle("Rumor Length (Original Vs Cleaned Vs Oversampled Vs Oversampled Cleaned")
sns.distplot(dfrl, label='Original Data')
sns.distplot(dfrcl, label='Original Cleaned Data')
sns.distplot(df['rumor_length'], label='Oversampled Data')
sns.distplot(df['rumor_cleaned_length'], label='Oversampled Cleaned Data')
fig.legend()

fig, axes = plt.subplots(figsize=(12,10))
fig.suptitle("Timeline Length (Original Vs Cleaned Vs Oversampled Vs Oversampled Cleaned")
sns.distplot(dftl, label='Original Data')
sns.distplot(dftcl, label='Original Cleaned Data')
sns.distplot(df['timeline_length'], label='Oversampled Data')
sns.distplot(df['timeline_cleaned_length'], label='Oversampled Cleaned Data')
fig.legend()

fig, axes = plt.subplots(figsize=(12,10))
fig.suptitle("Rumor Length (Original Vs Oversampled")
sns.distplot(dfrl, label='Original Data')
sns.distplot(df['rumor_length'], label='Oversampled Data')
fig.legend()

fig, axes = plt.subplots(figsize=(12,10))
fig.suptitle("Timeline Length (Original Vs Oversampled")
sns.distplot(dftl, label='Original Data')
sns.distplot(df['timeline_length'], label='Oversampled Data')
fig.legend()