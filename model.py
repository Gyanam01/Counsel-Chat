import pandas as pd

df = pd.read_csv(r'C:\Users\91858\Desktop\hackfest project\aisha\aisha\src\components\counselchat.csv')

col = ['questionText', 'topic']
df = df[col]

df.columns = ['questionText', 'topic']

df['category_id'] = df['topic'].factorize()[0]
from io import StringIO
category_id_df = df[['topic', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'topic']].values)

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')

features = tfidf.fit_transform(df.questionText).toarray()
labels = df.category_id

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(df['questionText'], df['topic'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

clf = MultinomialNB().fit(X_train_tfidf, y_train)

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split

model = LinearSVC()

X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

texts = ["hello"]
text_features = tfidf.transform(texts)



import pickle

pickle.dump(model, open('pmodel.pkl','wb'))

picklemodel = pickle.load(open('pmodel.pkl','rb'))
categ=picklemodel.predict(text_features)

print(id_to_category[categ[0]])
