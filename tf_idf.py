from __future__ import print_function
import glob, os
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Tf_idf_eval:
    def __init__(self,text_dir):
        self.text_dir = text_dir
        self.txtList = []
        self.pathList = []
        self.tfidf = None

    def load_tfidf(self):
        if self.tfidf != None:
            return
        
        path = self.text_dir
        os.chdir(path)
        
        import codecs
        for fl in glob.glob('*.txt'):
            self.pathList.append(fl)
            with open(fl) as f:
                self.txtList.append(f.read())
        
        corpus = np.asarray(self.txtList)
        self.vectorizer = CountVectorizer(min_df=1)
        bag = self.vectorizer.fit_transform(corpus)
        transformer = TfidfTransformer(smooth_idf=True)
        self.tfidf = transformer.fit_transform(bag)

    def get_tfidf(self,text):
        if self.tfidf == None:
            self.load_tfidf()
        return self.vectorizer.transform([text]).toarray()
    
    def compare(self,text1,text2):
        tf1 = self.get_tfidf(text1)
        tf2 = self.get_tfidf(text2)
        return cosine_similarity(tf1,tf2)[0][0]

txt_dir = 'peticoes_tokenized/'
test_text_1 = 'pagar debito acrescido custas processuais juros atualizacao monetaria'
test_text_2 = 'pagar debito acrescido bananas processuais juros atualizacao monetaria'
evaluator = Tf_idf_eval(txt_dir)
## For testing
#evaluator.load_tfidf()  # Can call 'get_tfidf' or 'compare' directly 
#print(get_tfidf(test_text_1))
##
print(test_text_1)
print(test_text_2)
print("Similaridade: "+str(evaluator.compare(test_text_1,test_text_2)))

## Scratch
#os.chdir(path)
#txtList = []
#pathList = []
#import codecs
#for fl in glob.glob('*.txt'):
#    pathList.append(fl)
#    #print(fl)
#    with open(fl) as f:
#        txtList.append(f.read())
#corpus = np.asarray(txtList)
#vectorizer = CountVectorizer(min_df=1)
#bag = vectorizer.fit_transform(corpus)
#transformer = TfidfTransformer(smooth_idf=True)
#tfidf = transformer.fit_transform(bag)
#d1 = 3
#d2 = 9
#t1 = vectorizer.transform(['pagar debito acrescido custas processuais juros atualizacao monetaria']).toarray()
#t2 = vectorizer.transform(['']).toarray()
#t1 = transformer.transform(t1).toarray()
#t2 = transformer.transform(t2).toarray()
#print(t)
##

