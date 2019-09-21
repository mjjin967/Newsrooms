import os
import sys
import xlrd
import docx
import sqlite3
# import nltk
# import sklearn


# import numpy as np

# from collections import Counter

# # Downloads the NLTK stopword corpus if not already downloaded
# try:
# 	nltk.data.fine('corpora/stopwords')
# except LookupError:
# 	nltk.download('stopwords')

# from nltk.corpus import stopwords
# from nltk.stem import SnowballStemmer
# from nltk.tokenize import RegexpTokenizer

# # sklearn modules for data processing
# from sklearn.preprocessing import LabelEncoder
# from sklearn.model_selection import train_test_split

# # sklearn modules for LSA
# from sklearn.decomposition import TruncatedSVD

# # sklearn modules for classification
# from sklearn.svm import SVC
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.neural_network import MLPClassifier

# # sklearn modules for clustering
# from sklearn.cluster import KMeans

# def process_document(text):
# 	"""
# 	Processes a text document by coverting all words to lower case,
# 	tokenizing, removing all non-alphabetical characters,
# 	and stemming each word.
# 	Args:
# 		text: A string of the text of a single document.
# 	Returns:
# 		A list of processed words from the document.
# 	"""
# 	# Convert words to lower case
# 	text = text.lower()

# 	# Tokenize corpus and remove all non-alphabetical characters
# 	tokenizer = RegexpTokenizer(r'\w+')
# 	tokens = tokenizer.tokenize(text)

# 	# Remove stopwords
# 	stop_words = stopwords.words('english')
# 	set_stopwords = set(stop_words)
# 	stopwords_removed = [token for token in tokens if not token in set_stopwords]

# 	# Stem words
# 	stemmer = SnowballStemmer('english')
# 	stemmed = [stemmer.stem(word) for word in stopwords_removed]

# 	# Return list of processed words
# 	return stemmed


def read_data(data_dir):
	"""
	Preprocesses all of the text documents in a directory.
	Args:
		data_dir: the directory to the data to be processed
	Returns:
		1. A mapping from zero-indexed document IDs to a bag of words
		(mapping from word to the number of times it appears)
		2. A mapping from words to the number of documents it appears in
		3. A mapping from words to unique, zero-indexed integer IDs
		4.  A mapping from document IDs to labels (politics, entertainment, tech, etc)
	"""
	document_titles = os.listdir(data_dir)
	articles = []
	for filename in document_titles:
		doc = docx.Document(data_dir+'/'+filename)
		fullText = []
		for para in doc.paragraphs:
			if len(para.text) != 0:
				fullText.append(para.text)
		articles.append('\n'.join(fullText))
# 	documents = {} # Mapping from document IDs to a bag of words
# 	document_word_counts = Counter() # Mapping from words to number of documents it appears in
# 	word_ids = {} # Mapping from words to unique integer IDs
# 	labels = {} # Mapping from document IDs to labels

# 	doc_id = 0
# 	word_id = 0
# 	for filename in os.listdir(data_dir):
# 		filepath = os.path.join(data_dir, filename)

# 		with open(filepath, 'r', errors='ignore') as f:
# 			doc = process_document(f.read())
# 			bag_of_words = {}
# 			# update documents, document_word_counts, word_ids, and labels
# 			for word in doc:
# 				if word in bag_of_words.keys():
# 					bag_of_words[word] += 1
# 				else:
# 					bag_of_words[word] = 1

# 				if word not in word_ids:
# 					word_ids[word] = word_id
# 					word_id += 1

# 			documents[doc_id] = bag_of_words
# 			document_word_counts += bag_of_words
# 			doc_id += 1
# 			labels[doc_id] = filename[0]
# 	return documents, document_word_counts, word_ids, labels


# def lsa(documents, document_word_counts, word_ids, num_topics=100, topics_per_document=3):
# 	"""
# 	Implements the LSA (Latent Semantic Analysis) algorithm
# 	to perform topic modeling.
# 	Args:
# 		documents: A mapping from zero-indexed document IDs to a bag of words
# 		document_word_counts: A mapping from words to the number of documents it appears in
# 		word_ids: A mapping from words to unique, zero-indexed integer IDs
# 	Returns:
# 		A dictionary that maps document IDs to a list of topics.
# 	"""
# 	# TODO: find the number of documents and words
# 	num_documents = len(documents)
# 	num_words = len(word_ids)
# 	tf_idf = np.zeros([num_documents, num_words])
# 	# tf score: num of times a word appears in a document / total num of words in document
# 	# idf score: log(total num of documents / num of documents containing word)

# 	# TODO: calculate the values in tf_idf
# 	for i in range(num_documents): # for each document
# 		bag_of_words = documents[i] # word -> num of times it appears in this ONE document
# 		total_words = 0 # in this particular document
# 		for word in bag_of_words.keys():
# 			word_id = word_ids[word]
# 			tf_idf[i][word_id] = np.log(num_documents/float(document_word_counts[word])) # how many times this word appears in this doc
# 			frequency = bag_of_words[word]
# 			tf_idf[i][word_id] *= frequency
# 			total_words += frequency
# 		tf_idf[i] /= total_words

# 	# Rows represent documents and columns represent topics
# 	document_topic_matrix = TruncatedSVD(n_components=num_topics, random_state=0).fit_transform(tf_idf)
# 	top_3_topics = {}
# 	for i in range(len(document_topic_matrix)):
# 		row = document_topic_matrix[i]
# 		argmaxes = np.argsort(row)[-topics_per_document:]
# 		top_3_topics[i] = argmaxes

# 	# TODO: return a dictionary that maps document IDs to a list of each one's top 3 topics
# 	return top_3_topics


# def classify_documents(topics, labels):
	# """
	# Classifies documents based on their topics.
	# Args:
	# 	topics: a dictionary that maps document IDs to topics.
	# 	labels: labels for each of the test files
	# Returns:
	# 	The score of each of the classifiers on the test data.
	# """

	# def classify(classifier):
	# 	"""
	# 	Trains a classifier and tests its performance.
	# 	NOTE: since this is an inner function within
	# 	classify_documents, this function will have access
	# 	to the variables within the scope of classify_documents,
	# 	including the train and test data, so we don't need to pass
	# 	them in as arguments to this function.
	# 	Args:
	# 		classifier: an sklearn classifier
	# 	Returns:
	# 		The score of the classifier on the test data.
	# 	"""
	# 	# fit the classifier on X_train and y_train
	# 	# and return the score on X_test and y_test
	# 	return classifier.fit(X_train, y_train).score(X_test, y_test)


	# # use topics and labels to create X and y
	# X = np.array(list(topics.values()))
	# y = np.array(list(labels.values()))

	# # use label_encoder to transform y
	# label_encoder = LabelEncoder()
	# y = label_encoder.fit_transform(y)

	# # modify the call to train_test_split to use
	# # 90% of the data for training and 10% for testing.
	# # Make sure to also shuffle and set a random state of 0!
	# X_train, X_test, y_train, y_test = train_test_split(
	# 	X,
	# 	y,
	# 	train_size=0.9,
	# 	test_size=0.1,
	# 	random_state=0,
	# 	shuffle=True
	# )


	# # create a KNeighborsClassifier that uses 3 neighbors to classify
	# knn = KNeighborsClassifier(n_neighbors=3)
	# knn_score = classify(knn)

	# # create a DecisionTreeClassifier with random_state=0
	# decision_tree = DecisionTreeClassifier(random_state=0)
	# decision_tree_score = classify(decision_tree)

	# # create an SVC with random_state=0
	# svm = SVC(random_state=0)
	# svm_score = classify(svm)

	# # create an MLPClassifier with random_state=0
	# mlp = MLPClassifier(random_state=0)
	# mlp_score = classify(mlp)

	# return knn_score, decision_tree_score, svm_score, mlp_score


def cluster_documents(topics, num_clusters=4):
	"""
	Clusters documents based on their topics.
	Args:
		document_topics: a dictionary that maps document IDs to topics.
	Returns:
		1. the predicted clusters for each document. This will be a list
		in which the first element is the cluster index for the first document
		and so on.
		2. the centroid for each cluster.
	"""
	pass
# 	k_means = KMeans(n_clusters=num_clusters, random_state=0)

# 	# Use k_means to cluster the documents and return the clusters and centers
# 	X = np.array(list(topics.values()))

# 	return k_means.fit_predict(X), k_means.cluster_centers_


# def plot_clusters(document_topics, clusters, centers):
# 	"""
# 	Uses matplotlib to plot the clusters of documents
# 	Args:
# 		document_topics: a dictionary that maps document IDs to topics.
# 		clusters: the predicted cluster for each document.
# 		centers: the coordinates of the center for each cluster.
# 	"""
# 	topics = np.array([x for x in document_topics.values()])

# 	ax = plt.figure().add_subplot(111, projection='3d')
# 	ax.scatter(topics[:, 0], topics[:, 1], topics[:, 2], c=clusters, alpha=0.3) # Plot the documents
# 	ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='black', alpha=1) # Plot the centers

# 	plt.tight_layout()
# 	plt.show()


def main(data_dir):
	"""
	This runs the program!
	Args:
		data_dir: the path to the BBC dataset directory.
	"""
	# Read in the data
	read_data(data_dir)
	# documents, document_word_counts, word_ids, labels = read_data(data_dir)

	# # Perform LSA
	# topics = lsa(documents, document_word_counts, word_ids)

	# # Classify the data
	# knn_score, decision_tree_score, svm_score, mlp_score = classify_documents(topics, labels)

	# print('\n===== CLASSIFIER PERFORMANCE =====')
	# print('K-Nearest Neighbors Accuracy: %.3f' % knn_score)
	# print('Decision Tree Accuracy: %.3f' % decision_tree_score)
	# print('SVM Accuracy: %.3f' % svm_score)
	# print('Multi-Layer Perceptron Accuracy: %.3f' % mlp_score)
	# print('\n')

	# # Cluster the data
	# clusters, centers = cluster_documents(topics)
	# plot_clusters(topics, clusters, centers)

# Run using 'python nlp.py' or 'python nlp.py <PATH_TO_BBC_DIRECTORY>'
# to manually specify the path to the data.
# This may take a little bit of time (~30-60 seconds) to run.
if __name__ == '__main__':
	# data_dir = '/course/cs1951a/pub/nlp/bbc/data' if len(sys.argv) == 1 else sys.argv[1]
	data_dir = os.path.join(os.getcwd(), '../doc_files')
	main(data_dir)