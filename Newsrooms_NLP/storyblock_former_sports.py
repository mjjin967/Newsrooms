from os import getcwd,listdir
from os.path import isfile, isdir, join
import sqlite3
import gensim
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
import collections



def taggedDoc_converter(tokens_only=False):
	conn = sqlite3.connect('bbc_sport.db')
	cur = conn.cursor()
	cur.execute('SELECT category,content from sports')
	data = cur.fetchall()
	conn.commit()
	conn.close()
	tagged_docs = []
	labels = []
	for i,text in enumerate(data):
		line = text[1]
		labels.append(text[0])
		if tokens_only:
			tagged_docs.append(gensim.utils.simple_preprocess(line))

		else:
			tagged_docs.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i]))
	return tagged_docs,labels

def kmeans_clustering(X,k):
	model = KMeans(n_clusters=k).fit(X)
	return model.cluster_centers_, model.predict(X), model.labels_


def doc2vec_tester():
	# content_train, content_test, y_train_str, y_test_str = train_test_split(contents,labels,test_size=0.20,random_state=1000)
	train_corpus, labels = taggedDoc_converter()

	model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
	# Build a vocabulary
	model.build_vocab(train_corpus)
	model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
	doc_embeddings = []
	for doc_id in range(len(train_corpus)): # for each document in testing corpus
		if len(train_corpus[doc_id].words) < 1:
			continue
		inferred_vector = model.infer_vector(train_corpus[doc_id].words)
		doc_embeddings.append(inferred_vector)
	cluster_centers, centroid_indices,numeric_labels = kmeans_clustering(doc_embeddings,5)

	label_tuples = [str([labels[i],numeric_labels[i]]) for i in range(len(labels))]
	counter_obj = collections.Counter(label_tuples)
	for item in counter_obj.items():
		print(item)


def main():
	curr_dir = getcwd()
	sql_directory = join(curr_dir, 'bbc_sport.db')
	doc2vec_tester()


if __name__ == '__main__':
	main()