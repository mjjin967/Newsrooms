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
	conn = sqlite3.connect('bbc_sport.db')
	cur = conn.cursor()
	cur.execute('SELECT title,content FROM sports limit 1000;')
	data = cur.fetchall()
	contents = []
	titles = []
	# published = []
	for row in data:
		titles.append(row[0])
		contents.append(row[1])
		# published.append(row[3])
	return titles,contents#,published

def retrieve_data():
	stoplist = set('for a of the and to in'.split())
	titles,documents = fetch_data()
	print("fetched data")
	texts = find_entity(documents)
	print("found entities")
	# print(texts[0:5])
	print("forming dictionary and corpus")
	dictionary = corpora.Dictionary(texts)

	corpus = [dictionary.doc2bow(text) for text in texts]
	# print(corpus[:10])
	return corpus,dictionary,titles#,entity_types#,documents,published

def find_entity(articles):
	nlp = en_core_web_sm.load()
	entities = []
	entity_types = []
	valid_entity_set = set(['PERSON', 'NORP', 'GPE', 'LOC', 'ORG', 'WORK_OF_ART', 'EVENT', 'LAW', 'FAC'])
	# set_to_check = set(['WORK_OF_ART', 'EVENT', 'LAW', 'FAC','ORDINAL'])
	for i,sentence in enumerate(articles):
		doc = nlp(str(sentence))
		entities += [[X.text for X in doc.ents if X.label_ in valid_entity_set]]
		# entity_types += [(X.text,X.label_) for X in doc.ents if X.label_ in set_to_check]
		# print(entity_types)
	# print(set(entity_types))
	return entities#,set(entity_types)

def train_tfidf_model():
	corpus,dictionary,titles = retrieve_data()
	# first, construct tfidf
	print("tfidf")
	tfidf = TfidfModel(corpus) # initialize model
	corpus_tfidf = tfidf[corpus]
	# print(entity_types)
	# print(tfidf)
	# print(corpus_tfidf)
	return corpus_tfidf,dictionary,titles

def load_and_cluster(k):
	corpus_tfidf,dictionary,titles = train_tfidf_model()
	lsi = LsiModel(corpus_tfidf,id2word=dictionary,num_topics=k)
	corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
	X = [[row[1] for row in document] for document in corpus_lsi]
	print("Kmeans")
	print([len(row) for row in X if len(row) != k])
	np_array = np.array([np.array(row) for row in X if len(row) == k])
	print(np_array.shape)

	model = KMeans(n_clusters=k).fit(np_array)
	return titles,model.cluster_centers_, model.labels_
	


def main():
	titles,cluster_centers,labels = load_and_cluster(300)
	# print("titles")
	# print(titles)
	print("cluster centers")
	print(cluster_centers.shape)
	# print(cluster_centers)
	print("labels")
	# print(labels)
	print(labels.shape)
	label_dict = dict()
	for i,label in enumerate(labels):
		if label in label_dict:
			label_dict[label].append(titles[i])
		else:
			label_dict[label] = [titles[i]]
	for k in label_dict.keys():
		print(k)
		for title in label_dict[k]:
			print(title)
		print('\n')




if __name__ == '__main__':
	main()