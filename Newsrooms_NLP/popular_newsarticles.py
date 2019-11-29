from os import getcwd,listdir
from os.path import isfile, isdir, join
import shutil
import json
import sqlite3
import pandas as pd
from gensim.models import LsiModel,TfidfModel
from gensim import corpora,similarities
# from gensim.test.utils import common_corpus,simple_process
import gensim
from gensim.test.utils import get_tmpfile
from collections import defaultdict
from sklearn.cluster import KMeans
import numpy as np
import collections
import pickle
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def store_in_db(data_directory):
	print("Reading in data...")
	dirs = [f for f in listdir(data_directory) if isdir(join(data_directory,f))]
	full_dirs = [join(data_directory,f) for f in dirs]
	sql_dir = 'popular_news.db'
	conn = sqlite3.connect(sql_dir)
	cur = conn.cursor()
	sql_command = """
	DROP TABLE IF EXISTS popular_news;
	CREATE TABLE popular_news (
		id INTEGER,
		title VARCHAR,
		content VARCHAR,
		published_date VARCHAR,
		news_site VARCHAR,
		site_section VARCHAR,
		PRIMARY KEY (id)
	);
	"""
	cur.executescript(sql_command)
	conn.commit()
	unique_id = 0
	for full_dir in full_dirs:
		print(full_dir)

		jsons = listdir(full_dir) # all the json files
		added_counter = 0; failed_counter = 0;
		for json_file in jsons:
			json_filename = json_file
			full_json_path = join(full_dir,json_file)
			try:
				with open(full_json_path,'r') as json_file:
					data_json = json.load(json_file)
					# print(type(data_json))
					# json_num = int(json_filename.split("_")[1].split(".")[0])
					title = data_json['title']
					content = data_json['text']
					published_date = data_json['published']
					news_site = data_json['thread']['site']
					site_section = data_json['thread']['site_section']
					added_counter += 1
					cur.execute('INSERT INTO popular_news (id, title, content, published_date, news_site, site_section) \
						VALUES (?,?,?,?,?,?);', (unique_id, title, content, published_date, news_site, site_section))
					conn.commit()
					unique_id += 1
			except OSError as e:
				print(e)
				failed_counter += 1
				break
			if added_counter % 10000 == 0:
				print("success")
				print(added_counter)
			if failed_counter > 0 and failed_counter % 10000 == 0:
				print("failed")
				print(failed_counter)
		
		print("successful: " + str(added_counter))
		print("failed on: " + str(failed_counter))
		print("After moving: " + str(len(listdir(full_dir))))
	conn.close()
	print("END of store_in_db\n")

def not_exist(field,data_list):
	if (field not in data_list.keys() or len(data_list[field]) == 0):
		return True
	return False

def fetch_data():
	conn = sqlite3.connect('popular_news.db')
	cur = conn.cursor()
	cur.execute('SELECT * FROM popular_news order by published_date limit 10000;')
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

def kmeans_clustering(X,k):
	model = KMeans(n_clusters=k).fit(X)
	pkl_filename = "pickle_model.pkl"
	with open(pkl_filename, 'wb') as file:
	    pickle.dump(model, file)
	# return model.cluster_centers_, model.predict(X), model.labels_

def retrieve_data():
	stoplist = set('for a of the and to in'.split())
	# entries = fetch_data(0,10)
	titles,documents,published = fetch_data()
	texts = [[word for word in document.lower().split() if word not in stoplist] for document in documents]
	frequency = defaultdict(int)
	for text in texts:
		for token in text:
			frequency[token] += 1
	texts = [[token for token in text if frequency[token] > 4] for text in texts]
	dictionary = corpora.Dictionary(texts)
	corpus = [dictionary.doc2bow(text) for text in texts]
	return corpus,dictionary,titles#,documents,published

def train_tfidf_model():
	corpus,dictionary,titles = retrieve_data()
	# first, construct tfidf
	print("tfidf")
	tfidf = TfidfModel(corpus) # initialize model
	corpus_tfidf = tfidf[corpus]
	return corpus_tfidf,dictionary,titles

def save_lsi_model(corpus_tfidf,dictionary):
	# apply transformation to whole corpus
	print("lsi model")
	lsi = LsiModel(corpus_tfidf,id2word=dictionary,num_topics=3000) # initialize LSI transformation
	tmp_fname = get_tmpfile("lsi.model")
	print("saving tmp file")
	lsi.save(tmp_fname)
	return tmp_fname

def load_and_cluster():
	corpus_tfidf,dictionary,titles = train_tfidf_model()
	# fname = save_lsi_model(corpus_tfidf,dictionary)
	print("fname")
	fname = "/var/folders/ft/jlv83lxd58zb3v6bjqtzlr0c0000gn/T/lsi.model"
	print(fname)
	lsi = LsiModel.load(fname)
	print("lsi corpus")
	corpus_lsi = lsi[corpus_tfidf] # create a double wrapper over the original corpus: bow->tfidf->fold-in-lsi
	X = [[row[1] for row in document] for document in corpus_lsi]
	print("Kmeans")
	print([row for row in X if len(row) != 3000])
	np_array = np.array([np.array(row) for row in X if len(row) == 3000])
	print(type(np_array))
	print(type(np_array[0]))
	print(np.unique([len(row) for row in np_array]))
	kmeans_clustering(np_array,1000)
	with open("pickle_model.pkl", 'rb') as file:
    	kmeans_model = pickle.load(file)
    cluster_centers = kmeans_model.cluster_centers_
    numeric_labels = kmeans_model.labels_
    #, model.predict(X), model.labels_
	print(np.unique(numeric_labels))
	for i in range(1000):
		if i % 100 == 0:
			print(i)
			for j,label in enumerate(numeric_labels):
				if label == i:
					print(titles[j])
	# print(row[1] for row in corpus_lsi[0])
	# similarity query
	# vec_bow = dictionary.doc2bow(documents[50].lower().split())
	# vec_lsi = lsi[vec_bow]
	# index = similarities.MatrixSimilarity(lsi[corpus])
	# sims = index[vec_lsi]
	# sims = sorted(enumerate(sims), key=lambda item: -item[1])[:20]
	# print(titles[50])
	# for row in sims:
	# 	i = row[0]
	# 	print(row,titles[i],published[i])



def main():
	curr_dir = getcwd()
	data_directory = join(curr_dir, '../Newsrooms_datasets/Popular_News_Webhose/valid_news')
	# tmp_fname,corpus_tfidf = train_model()
	load_and_cluster()



if __name__ == '__main__':
	main()



