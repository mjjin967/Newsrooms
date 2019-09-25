import os
import sqlite3
import nltk

# Downloads the NLTK stopword corpus if not already downloaded
try:
# <<<<<<< master
	nltk.download('stopwords')
except Exception as e:
	print(e)
    #nltk.data.fine('corpora/stopwords')
# =======
# 	nltk.data.fine('corpora/stopwords')
# except LookupError:
# 	nltk.download('stopwords')
# >>>>>>> master

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import RegexpTokenizer


def process_document(text):
	"""
	Processes a text document by coverting all words to lower case,
	tokenizing, removing all non-alphabetical characters,
	and stemming each word.
	Args:
		text: A string of the text of a single document.
	Returns:
		A list of processed words from the document.
	"""
	# Convert words to lower case
	text = text.lower()

# <<<<<<< master
	# Tokenize corpus and remove all non-alphabetical characters
	tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(text)

	# Remove stopwords
	stop_words = nltk.corpus.stopwords.words('english')
# =======
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
# >>>>>>> master
	set_stopwords = set(stop_words)
	stopwords_removed = [token for token in tokens if not token in set_stopwords]

	# Stem words
# <<<<<<< master
	stemmer = nltk.stem.SnowballStemmer('english')
# =======
# 	stemmer = SnowballStemmer('english')
# >>>>>>> master
	stemmed = [stemmer.stem(word) for word in stopwords_removed]

	# Return list of processed words
	return stemmed

def load_articles(data_dir):
	"""
	Returns a hashmap key: unique id, val: title
	"""
	conn = sqlite3.connect(data_dir)
	cur = conn.cursor()
	categories = dict()
	cur.execute('SELECT title,content from articles')
	data = cur.fetchall()
	print("length of data is " + str(len(data)))
	res=[data[i] for i in range(len(data)) if len(data[i]) != 0]
	return res

def return_json(data_dir):
	"""
	"""
	conn = sqlite3.connect('articles_db.db')
	cur = conn.cursor()
	categories = dict()
	cur.execute('SELECT id,title,content,category from articles')
	data = cur.fetchall()
	for entry in data:
		cat_num = entry[3]
		if cat_num in categories.keys():
			categories[cat_num].append({'id': entry[0], 'title': entry[1], 'content': entry[2]})
		else:
			categories[cat_num] = [{'id': entry[0], 'title': entry[1], 'content': entry[2]}]
	return categories


def classify_documents(topics, labels):
	pass


def cluster_documents():
	pass



# def main(data_dir):
# 	pass
# 	# corpus = load_articles(data_dir) # list of tuples [title, article]
# # 	labels = load_labels(data_dir) # hashmap of key: unique doc id, val: title
# #     return [corpus, labels]


# # Run using 'python nlp.py' or 'python nlp.py <PATH_TO_BBC_DIRECTORY>'
# # to manually specify the path to the data.
# # This may take a little bit of time (~30-60 seconds) to run.
# if __name__ == '__main__':
# 	# data_dir = '/course/cs1951a/pub/nlp/bbc/data' if len(sys.argv) == 1 else sys.argv[1]
# 	data_dir = os.path.join(os.getcwd(), 'articles_db.db')
# 	main(data_dir)