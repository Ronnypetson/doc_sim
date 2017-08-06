import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Hiperparameters: size of grams

def tf_idf(counts):
    transformer = TfidfTransformer(smooth_idf=True)
    tfidf = transformer.fit_transform(counts)
    return tfidf.toarray()

#corpus = [
#    'This is the first document.',
#    'This is the second second document.',
#    'And the third one.',
#    'Is this the first document?',
#]

path = 'texts/got.txt'

import codecs
data = codecs.open(path,encoding='ISO-8859-1').read().lower()

txtList = data.split(".")
corpus = np.asarray(txtList)
# analyze = vectorizer.build_analyzer(), analyze("This is a text document to analyze.")
# vectorizer.get_feature_names(), X.toarray(), vectorizer.vocabulary_.get('document')
# vectorizer.transform(['Something completely new.']).toarray()
# Bag of 2-grams also available

vectorizer = CountVectorizer(min_df=1)
vectorizer.fit_transform(corpus)

def cos_sim(f1,f2):
    c1 = vectorizer.transform([f1]).toarray()
    c2 = vectorizer.transform([f2]).toarray()
    tf1 = tf_idf(c1)
    tf2 = tf_idf(c2)
    return cosine_similarity(tf1,tf2)

f1 = "the water pipe is off since friday"
f2 = "No water since friday. We need a pumbler to fix the water pipe"
print("Comparando:\n\n"+f1+"\n\n"+f2+"\n")
print("Similaridade: "+str(cos_sim(f1,f2)[0][0]))

