import gensim as gs

w2v = gs.models.KeyedVectors.load_word2vec_format('word2vec/GoogleNews-vectors-negative300.bin/data', binary=True)
# w2v.similarity('man', 'human')
coded = gs.models.Word2Vec()
print(w2v.vocab["man"].index())