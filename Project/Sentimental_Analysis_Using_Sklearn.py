#%%
import numpy as np
import pandas as pd
import re
import nltk
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
# %%
df = pd.read_csv('movie_data.csv')
df.head(n=5)
# %%
count = CountVectorizer()
docs = np.array([
    'The sun is shining',
    'The weather is sweet',
    'The sun is shining,the weather is sweet,and one and one is two'
])
bag = count.fit_transform(docs)
# %%
print(count.vocabulary_)
print(bag.toarray())
# %%
np.set_printoptions(precision=2)
# %%
tfidf = TfidfTransformer(use_idf=True,norm='l2',smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())
# %%
tf_is = 3
n_docs = 3
idf_is = np.log((n_docs+1)/(3+1))
tfidf_is = tf_is*(idf_is+1)
print(tfidf_is)
# %%
tfidf = TfidfTransformer(use_idf=True,norm=None,smooth_idf=True)
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
print(raw_tfidf)
l2_tfidf = raw_tfidf/np.sqrt(np.sum(raw_tfidf**2))
print(l2_tfidf)
# %%
df.loc[0,'review'][-50:]
# %%
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    return text
# %%
preprocessor(df.loc[0,'review'][-50:])
# %%
preprocessor('</a>This:)is :(a test :-)!')
# %%
df['review'] = df['review'].apply(preprocessor)
# %%
porter = PorterStemmer()
def tokenizer(text):
    return text.split()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

tokenizer('runners like running and thus they run')
tokenizer_porter('runners like running and thus they run')
# %%
nltk.download('stopwords')
# %%
stop = stopwords.words('english')
[w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]if w not in stop]
# %%
X_train = df.loc[:25000,'review'].values
y_train = df.loc[:25000,'sentiment'].values
X_test = df.loc[25000:,'review'].values
y_test = df.loc[25000:,'sentiment'].values
print('Review: ',X_train[1:2],'\n')
print('Sentiment: ',y_train[1])
# %%
tfidf = TfidfVectorizer(strip_accents=None,lowercase=False,preprocessor=None)
param_grid = [{'vect__ngram_range': [(1,1)],
             'vect__stop_words': [stop,None],
             'vect__tokenizer':[tokenizer,tokenizer_porter],
             'clf__penalty':['l2'],
             'clf__C':[1.0,10.0,100.0]},
             {'vect__ngram_range':[(1,1)],
               'vect__stop_words':[stop,None],
               'vect__tokenizer':[tokenizer,tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty':['l2'],
               'clf__C':[1.0,10.0,100.0]}]
lr_tfidf = Pipeline([('vect',tfidf),('clf',LogisticRegression(random_state=0))])
gs_lr_tfidf = GridSearchCV(lr_tfidf,param_grid,scoring='accuracy',cv=5,verbose=1,n_jobs=-1)
# %%
import pickle
with open('saved_model.sav','rb') as f:
    gs_lr_tfidf=pickle.load(f)
# %%
print(gs_lr_tfidf.best_params_)
print(gs_lr_tfidf.best_score_)
clf = gs_lr_tfidf.best_estimator_
print(clf.score(X_test,y_test))
# %%
