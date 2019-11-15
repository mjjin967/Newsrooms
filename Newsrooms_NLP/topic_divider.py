from os import getcwd,listdir
from os.path import isfile, isdir, join
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import ensemble

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def store_data(data_dir, sql_dir):
	print("Connecting to the database...")
	conn = sqlite3.connect(sql_dir)
	cur = conn.cursor()
	sql_command = """
	DROP TABLE IF EXISTS sports;
	CREATE TABLE sports (
		id INTEGER,
		title VARCHAR,
		content VARCHAR,
		category VARCHAR,
		PRIMARY KEY (id)
	);
	"""
	cur.executescript(sql_command)
	conn.commit()
	print("Connected to the database.")
	print("Reading in the data...")
	categories = [f for f in listdir(data_dir)]
	unique_id = 0
	import codecs
	for category in categories:
		dir_path = join(data_dir,category)
		if isdir(dir_path):
			files = listdir(dir_path)
			for file in files:
				with codecs.open(join(dir_path,file), 'rb') as f:
					title = f.readline().rstrip()
					content = f.read().rstrip()
					cur.execute('INSERT INTO sports (id,title,content,category) VALUES (?,?,?,?);',\
						(unique_id,title,content,category))
					conn.commit()
					unique_id += 1
	conn.close()
	print("Data successfully stored in sql database.")

def dataset_divider(sql_dir, vectorizer):
	# print("Dividing the dataset...")
	conn = sqlite3.connect(sql_dir)
	cur = conn.cursor()
	cur.execute('SELECT title,content,category from sports')
	result = cur.fetchall()
	conn.commit()
	conn.close()
	df = pd.DataFrame(iter(result),columns=['title','content','category'])
	# print(df.category.value_counts())
	contents = df['content'].values
	y = df['category'].values
	content_train, content_test, y_train_str, y_test_str = train_test_split(contents, y, test_size=0.20, random_state=1000)

	# print("Fitting the model")
	vectorizer.fit(content_train)
	X_train = vectorizer.transform(content_train)
	X_test = vectorizer.transform(content_test)
	encoder = preprocessing.LabelEncoder()
	y_train = encoder.fit_transform(y_train_str)
	y_test = encoder.fit_transform(y_test_str)
	return X_train, X_test, y_train, y_test


def count_vectorizer(sql_dir):
	# print("Using CountVectorizer")
	vectorizer = CountVectorizer(encoding='latin-1')
	return dataset_divider(sql_dir,vectorizer)

def tfidf_vectorizer_word(sql_dir):
	# word level tf-idf
	tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000,encoding='latin-1')
	return dataset_divider(sql_dir,tfidf_vect)

def tfidf_vectorizer_ngram(sql_dir):
	# ngram level tf-idf 
	tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000,encoding='latin-1')
	# tfidf_vect_ngram.fit(sentences_train)
	# X_train_tfidf_ngram =  tfidf_vect_ngram.transform(sentences_train)
	# X_test_tfidf_ngram =  tfidf_vect_ngram.transform(sentences_test)
	return dataset_divider(sql_dir,tfidf_vect_ngram)

def tfidf_vectorizer_chars(sql_dir):
	# characters level tf-idf
	tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000,encoding='latin-1')
	# tfidf_vect_ngram_chars.fit(sentences_train)
	# X_train_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(sentences_train) 
	# X_test_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(sentences_test)
	return dataset_divider(sql_dir,tfidf_vect_ngram_chars)

def train_model(classifier, feature_vector_train, label, feature_vector_valid, valid_y, is_neural_net=False):
	# fit the training dataset on the classifier
	# print("Training...")
	classifier.fit(feature_vector_train, label)

	# predict the labels on validation dataset
	# print("Predicting...")
	predictions = classifier.predict(feature_vector_valid)
	if is_neural_net:
		predictions = predictions.argmax( axis=-1)

	return classifier, metrics.accuracy_score(predictions, valid_y)

def train_and_test_balanced(sql_dir):
	X_train, X_test, y_train, y_test = count_vectorizer(sql_dir)
	X_train_tfidf, X_test_tfidf, y_train, y_test = tfidf_vectorizer_word(sql_dir)
	# X_train_tfidf, X_test_tfidf, y_train, y_test = tfidf_vectorizer_ngram(sql_dir)
	# X_train_tfidf, X_test_tfidf, y_train, y_test = tfidf_vectorizer_chars(sql_dir)

	# Naive Bayes on Count Vectors
	nb, accuracy = train_model(naive_bayes.MultinomialNB(), X_train, y_train, X_test, y_test)
	print("NB, Count Vectors: ", accuracy)

	# Naive Bayes on TF-IDF Vectors
	avg_accuracy = 0
	for i in range(10):
		nb_tfidf, accuracy = train_model(naive_bayes.MultinomialNB(), X_train_tfidf, y_train, X_test_tfidf, y_test)
		avg_accuracy += accuracy
	avg_accuracy /= 10
	print("NB, TF-IDF Vectors: ", avg_accuracy)

	# Linear Classifier on Count Vectors
	lr, accuracy = train_model(linear_model.LogisticRegression(), X_train, y_train, X_test, y_test)
	print ("LR, Count Vectors: ", accuracy)

	# Linear Classifier on TF-IDF Vectors
	avg_accuracy = 0
	for i in range(10):
		lr_tfidf, accuracy = train_model(linear_model.LogisticRegression(), X_train_tfidf, y_train, X_test_tfidf, y_test)
		avg_accuracy += accuracy
	avg_accuracy /= 10
	print ("LR, TF-IDF Vectors: ", accuracy)

	# RF on Count Vectors
	rf, accuracy = train_model(ensemble.RandomForestClassifier(), X_train, y_train, X_test, y_test)
	print("RF, Count Vectors: ", accuracy)

	# RF on TF-IDF Vectors
	avg_accuracy = 0
	for i in range(10):
		rf_tfidf, accuracy = train_model(ensemble.RandomForestClassifier(), X_train_tfidf, y_train, X_test_tfidf, y_test)
		avg_accuracy += accuracy
	avg_accuracy /= 10
	print("RF, TF-IDF Vectors: ", accuracy)

def train_and_test_unbalanced(sql_dir):
	X_train, X_test, y_train, y_test = count_vectorizer(sql_dir)
	X_train_tfidf, X_test_tfidf, y_train, y_test = tfidf_vectorizer_word(sql_dir)
	# X_train_tfidf, X_test_tfidf, y_train, y_test = tfidf_vectorizer_ngram(sql_dir)
	# X_train_tfidf, X_test_tfidf, y_train, y_test = tfidf_vectorizer_chars(sql_dir)

	# Naive Bayes on Count Vectors
	nb, accuracy = train_model(naive_bayes.ComplementNB(), X_train, y_train, X_test, y_test)
	print("NB, Count Vectors: ", accuracy)

	# Naive Bayes on TF-IDF Vectors
	nb_tfidf, accuracy = train_model(naive_bayes.ComplementNB(), X_train_tfidf, y_train, X_test_tfidf, y_test)
	print("NB, TF-IDF Vectors: ", accuracy)

	# Linear Classifier on Count Vectors
	lr, accuracy = train_model(linear_model.LogisticRegression(class_weight='balanced'), X_train, y_train, X_test, y_test)
	print ("LR, Count Vectors: ", accuracy)

	# Linear Classifier on TF-IDF Vectors
	lr_tfidf, accuracy = train_model(linear_model.LogisticRegression(class_weight='balanced'), X_train_tfidf, y_train, X_test_tfidf, y_test)
	print ("LR, TF-IDF Vectors: ", accuracy)

	# RF on Count Vectors
	rf, accuracy = train_model(ensemble.RandomForestClassifier(class_weight='balanced'), X_train, y_train, X_test, y_test)
	print("RF, Count Vectors: ", accuracy)

	# RF on TF-IDF Vectors
	rf_tfidf, accuracy = train_model(ensemble.RandomForestClassifier(class_weight='balanced'), X_train_tfidf, y_train, X_test_tfidf, y_test)
	print("RF, TF-IDF Vectors: ", accuracy)

def main():
	curr_dir = getcwd()
	data_directory = join(curr_dir, 'bbcsport')
	sql_directory = join(curr_dir, 'bbc_sport.db')
	# store_data(data_directory, sql_directory)
	# divide_dataset(sql_directory)
	# train_and_test_balanced(sql_directory)
	train_and_test_unbalanced(sql_directory)

if __name__ == '__main__':
	main()



