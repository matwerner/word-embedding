ARTICLES=10000
TOKENS=30000
DUMP_PATH="../datasets/ptwiki-20170820-pages-articles-parsed.json"
FULL_VOCAB_PATH="../datasets/ptwiki-20170820-sentences-doc10k.pickle"
LIMITED_VOCAB_PATH="../datasets/ptwiki-20170820-sentences-doc10k-vocab30k.pickle"
STOPWORDS="../datasets/stopwords.txt"

python3 preprocessing.py --n_articles $ARTICLES --n_tokens $TOKENS --wiki_dump $DUMP_PATH --full_vocab $FULL_VOCAB_PATH --limited_vocab $LIMITED_VOCAB_PATH --stopwords $STOPWORDS --verbose True

CORPUS_PATH='../datasets/ptwiki-20170820-sentences-doc10k-vocab30k.pickle'
EMBEDDING_PATH='../datasets/ptwiki-20170820-embedding-doc10k-vocab30k-embedding20.pickle'

#python3 word2vec.py --embedding 30 --annel 20000 --print 1000 --corpus_path $CORPUS_PATH --embedding_path $EMBEDDING_PATH


