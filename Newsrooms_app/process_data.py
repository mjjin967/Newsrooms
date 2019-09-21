import os
import sqlite3
import json

def return_json():
	"""
	"""
	conn = sqlite3.connect('comprehensive_articles_db.db')
	cur = conn.cursor()
	categories = dict()
	cur.execute('SELECT * from articles')
	data = cur.fetchall()
	for entry in data:
		_id,title,publisher,url,time_stamp,content,category = entry
		if category in categories.keys():
			categories[category].append({'id': str(_id), 'title': str(title), 'publisher': str(publisher),\
				'url': str(url), 'time_stamp': str(time_stamp), 'content': str(content)})
		else:
			categories[category] = [{'id': str(_id), 'title': str(title), 'publisher': str(publisher),\
				'url': str(url), 'time_stamp': str(time_stamp), 'content': str(content)}]
	json_dict = json.dumps(categories)
	print(json_dict)


if __name__ == '__main__':
	return_json()