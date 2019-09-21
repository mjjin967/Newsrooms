from os import listdir, getcwd
from os.path import isfile, join
import docx2txt
import sqlite3
import xlrd


def read_excel(excel_dir):
	wb = xlrd.open_workbook(excel_dir)
	sheet = wb.sheet_by_index(1)
	# titles = [sheet.cell_value(i,4) for i in range(2,sheet.nrows)]
	# categories = [sheet.cell_value(i,6) for i in range(2,sheet.nrows)]
	category_hashmap = dict()
	for i in range(2,sheet.nrows):
		title = sheet.cell_value(i,4)
		category = int(sheet.cell_value(i,6))
		category_hashmap[title] = category
	return category_hashmap

def store_data(data_dir, excel_dir):
	print("Reading in the excel spreadsheet...")
	category_hashmap = read_excel(excel_dir)
	articles = []
	print("Connecting to the database...")
	conn = sqlite3.connect('articles_categories_db_1.db')
	cur = conn.cursor()
	sql_command = """
	DROP TABLE IF EXISTS articles;
	CREATE TABLE articles (
		id INTEGER,
		title VARCHAR,
		content VARCHAR,
		category INTEGER,
		PRIMARY KEY (id)
	);
	"""
	cur.executescript(sql_command)
	conn.commit()
	print("Processing docx files...")
	onlyfiles = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
	unique_id = 0
	to_be_processed = []
	not_in_excel_sheet = []
	for f in listdir(data_dir):
		full_path = join(data_dir, f)
		if not isfile(full_path):
			continue
		if "MSNBC" in full_path:
			continue

		text = docx2txt.process(full_path)
		fullText = []
		
		for line in text.splitlines():
			if len(line) != 0:
				fullText.append(line)
		if "(" in f or ("Page" in fullText[0] and "of" in fullText[0]):
			if "Page" in fullText[0] and "of" in fullText[0]:
				num_articles = 1
			else:
				num_articles = int((f.split("(")[1]).split(")")[0])
			k = 0
			titles = []
			while k < len(fullText):
				if "Client/Matter" in fullText[k]:
					title = fullText[k-1].split('. ')[1]
					titles.append(title)
				k += 1
			i = 7
			title_ind = 0
			while i < len(fullText):
				para = fullText[i]
				if "Body" in para:
					article_body = []
					i += 1
					while fullText[i+1] != "Classification":
						article_body.append(fullText[i])
						i += 1
					if titles[title_ind] in category_hashmap.keys():
						sql_body = '\n'.join(article_body)
						category = category_hashmap[titles[title_ind]]
						cur.execute('INSERT INTO articles (id, title, content, category) VALUES (?,?,?,?);', (unique_id, titles[title_ind], sql_body, category))
						conn.commit()
						unique_id += 1 # 
						title_ind += 1
					else:
						not_in_excel_sheet.append(titles[title_ind])
						# print(f + ", " + titles[title_ind])
						title_ind += 1
						
				i += 1
		else:
			ind = 1
			while len(fullText[ind]) < 45:
				ind += 1
			sql_title = fullText[0]
			if sql_title in category_hashmap.keys():
				sql_body = '\n'.join(fullText[ind:])
				category = category_hashmap[sql_title]
				cur.execute('INSERT INTO articles (id, title, content, category) VALUES (?,?,?,?);', (unique_id, sql_title, sql_body, category))
				conn.commit()
				unique_id += 1
			else:
				not_in_excel_sheet.append(sql_title)
				# print(f + ", " + sql_title)
	conn.close()



def main(data_dir):
	"""
	This runs the program!
	Args:
		data_dir: the path to the docx files directory.
	"""
	# Read in the data
	data_directory = join(data_dir, '../doc_files')
	excel_directory = join(data_dir, '../article_categorization.xlsx')
	# read_excel(excel_directory)
	store_data(data_directory, excel_directory)


if __name__ == '__main__':
	# data_dir = join(getcwd(), '../doc_files')
	main(getcwd())

