import gensim
import os
import collections
import smart_open
import random
import sqlite3



def read_corpus(tokens_only=False):
    conn = sqlite3.connect('articles_db.db')
    cur = conn.cursor()
    cur.execute('SELECT category,content from articles')
    data = cur.fetchall()
    for i, text in enumerate(data):
        # print(text[1])
        line = text[1].encode()
        if tokens_only:
            yield gensim.utils.simple_preprocess(line)
        else:
            # For training data, add tags
            yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])


train_corpus = list(read_corpus())
test_corpus = list(read_corpus(tokens_only=True))

# instantiate a Doc2Vec Object
model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)

# Build a vocabulary
model.build_vocab(train_corpus)

model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

# assess the model
ranks = []
second_ranks = []
for doc_id in range(len(train_corpus)): # for each document in training corpus
    inferred_vector = model.infer_vector(train_corpus[doc_id].words)
    sims = model.docvecs.most_similar([inferred_vector], topn=len(model.docvecs))
    rank_list = [docid for docid, sim in sims]
    try:
        rank = rank_list.index(doc_id)
        if rank != 0:
        	print("Original article is: ")
        	print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
        	print("Article detected as most similar: ")
        	print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[0].words)))
        ranks.append(rank)
    except ValueError:
        print(doc_id)
    
    second_ranks.append(sims[1])

print(collections.Counter(ranks))

# print('Document ({}): «{}»\n'.format(doc_id, ' '.join(train_corpus[doc_id].words)))
# print(u'SIMILAR/DISSIMILAR DOCS PER MODEL %s:\n' % model)
# for label, index in [('MOST', 0), ('SECOND-MOST', 1), ('MEDIAN', len(sims)//2), ('LEAST', len(sims) - 1)]:
#     print(u'%s %s: «%s»\n' % (label, sims[index], ' '.join(train_corpus[sims[index][0]].words)))