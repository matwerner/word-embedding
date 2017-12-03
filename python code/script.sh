BASE_PATH="../datasets/ptwiki-20170820"

# Preprocessing

DOC_FORMAT="doc10k"

ARTICLES=10000
CORPORA_PATH="$BASE_PATH-pages-articles-parsed.json"
SENTENCES_PATH="$BASE_PATH-$DOC_FORMAT-sentences.pickle"
TOKENS_PATH="$BASE_PATH-$DOC_FORMAT-tokens.pickle"
STOPWORDS="../datasets/stopwords.txt"

python3 preprocessing.py preprocess --corpora $CORPORA_PATH --limit $ARTICLES --stopwords $STOPWORDS --sentences $SENTENCES_PATH --tokens $TOKENS_PATH --verbose True

VOCAB_FORMAT="vocab30k"

# Filter vocabulary

TOKENS=30000
FILTER_SENTENCES_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-sentences.pickle"
FILTER_TOKENS_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-tokens.pickle"-$VOCAB_FORMAT

python3 preprocessing.py filter --limit $TOKENS --raw_sentences $SENTENCES_PATH --sentences $FILTER_SENTENCES_PATH --tokens $FILTER_TOKENS_PATH --verbose True

# Word2Vec training

EMBEDDING_SIZE="embedding10"

RANDOM_SIZE=1000000
EMBEDDING=10
ANNEAL_RATE=0.8
ANNEAL_EVERY=10000
PRINT_EVERY=1000
SENTENCES_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-sentences.pickle"
TOKENS_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-tokens.pickle"
FREQUENCIES_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-frequencies.pickle"
W_IN_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-win.pickle"
W_OUT_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-wout.pickle"

#python3 word2vec.py --random $RANDOM_SIZE --embedding $EMBEDDING --anneal_rate $ANNEAL_RATE --anneal $ANNEAL_EVERY --print $PRINT_EVERY --sentences_path $SENTENCES_PATH  --tokens_path $TOKENS_PATH  --frequencies_path $FREQUENCIES_PATH --w_in_path $W_IN_PATH --w_out_path $W_OUT_PATH


