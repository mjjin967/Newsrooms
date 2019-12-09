import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import sqlite3
from gensim import corpora
from gensim.models import LsiModel,TfidfModel
import numpy as np
from sklearn.cluster import KMeans

def fetch_data():
	conn = sqlite3.connect('popular_news.db')
	cur = conn.cursor()
	cur.execute('SELECT * FROM popular_news order by published_date limit 1000;')
	# cur.execute('SELECT * FROM popular_news where id between %d and %d;'%(start,end))
	data = cur.fetchall()
	# print(data[:20])
	contents = []
	titles = []
	published = []
	for row in data:
		content = row[2]
		title = row[1]
		contents.append(content)
		titles.append(title)
		published.append(row[3])
	return titles,contents,published

def retrieve_data():
	stoplist = set('for a of the and to in'.split())
	titles,documents,published = fetch_data()
	texts = find_entity(documents)
	print("retrieve_data")
	print(texts[0:5])
	dictionary = corpora.Dictionary(texts)

	corpus = [dictionary.doc2bow(text) for text in texts]
	print(corpus[0])
	print(dictionary[0])
	return corpus,dictionary,titles#,documents,published

def find_entity(articles):
	nlp = en_core_web_sm.load()
	entities = []
	entity_types = []
	for i,sentence in enumerate(articles):
		doc = nlp(sentence)
		entities += [[X.text for X in doc.ents]]
	return entities

def train_tfidf_model():
	corpus,dictionary,titles = retrieve_data()
	# first, construct tfidf
	print("tfidf")
	tfidf = TfidfModel(corpus) # initialize model
	corpus_tfidf = tfidf[corpus]
	print(tfidf)
	print(corpus_tfidf)
	return corpus_tfidf,dictionary,titles

def load_and_cluster():
	corpus_tfidf,dictionary,titles = train_tfidf_model()
	lsi = LsiModel(corpus_tfidf,id2word=dictionary,num_topics=300)
	corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
	X = [[row[1] for row in document] for document in corpus_lsi]
	print("Kmeans")
	print([row for row in X if len(row) != 300])
	np_array = np.array([np.array(row) for row in X if len(row) == 300])
	print(type(np_array))
	print(type(np_array[0]))
	print(np.unique([len(row) for row in np_array]))
	print(np_array[0])
	model = KMeans(n_clusters=300).fit(np_array)
	return titles,model.cluster_centers_,None,None#, model.predict(X), model.labels_,titles
	


def main():
	titles,cluster_centers,indices,labels = load_and_cluster()
	# print(cluster_centers[:5])
	print(set(indices))
	print(set(labels))
	# indices shows which cluster it belongs to, labels shows 
	# titles is
	for i in enumerate(set(labels)):
		print(i)
		for j in labels:
			if i == j:
				print(titles[j])



if __name__ == '__main__':
	main()