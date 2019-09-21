import nltk

# Downloads the NLTK stopword corpus if not already downloaded
try:
	nltk.data.fine('corpora/stopwords')
except LookupError:
	nltk.download('stopwords')

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

	# Tokenize corpus and remove all non-alphabetical characters
	tokenizer = RegexpTokenizer(r'\w+')
	tokens = tokenizer.tokenize(text)

	# Remove stopwords
	stop_words = stopwords.words('english')
	set_stopwords = set(stop_words)
	stopwords_removed = [token for token in tokens if not token in set_stopwords]

	# Stem words
	stemmer = SnowballStemmer('english')
	stemmed = [stemmer.stem(word) for word in stopwords_removed]

	# Return list of processed words
	return stemmed