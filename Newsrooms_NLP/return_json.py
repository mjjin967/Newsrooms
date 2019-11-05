

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


def classify_documents():
	data = load_articles('articles_db')



def cluster_documents():
	pass

