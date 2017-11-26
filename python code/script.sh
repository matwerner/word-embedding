BASE_PATH="../datasets/ptwiki-20170820"

# Preprocssing

DOC_FORMAT="doc1kk"
VOCAB_FORMAT="vocab30k"

ARTICLES=1000000
TOKENS=30000
DUMP_PATH="$BASE_PATH-pages-articles-parsed.json"
FULL_SENTENCES_PATH="$BASE_PATH-sentences-$DOC_FORMAT.pickle"
FULL_TOKENS_PATH="$BASE_PATH-tokens-$DOC_FORMAT.pickle"
FULL_FREQUENCIES_PATH="$BASE_PATH-frequencies-$DOC_FORMAT.pickle"
FILTERED_SENTENCES_PATH="$BASE_PATH-sentences-$DOC_FORMAT-$VOCAB_FORMAT.pickle"
FILTERED_TOKENS_PATH="$BASE_PATH-tokens-$DOC_FORMAT-$VOCAB_FORMAT.pickle"
FILTERED_FREQUENCIES_PATH="$BASE_PATH-frequencies-$DOC_FORMAT-$VOCAB_FORMAT.pickle"
STOPWORDS="../datasets/stopwords.txt"

python3 preprocessing.py --n_articles $ARTICLES --n_tokens $TOKENS --wiki_dump $DUMP_PATH --stopwords $STOPWORDS --verbose True --full_sentences $FULL_SENTENCES_PATH  --full_tokens $FULL_TOKENS_PATH  --full_frequencies $FULL_FREQUENCIES_PATH  --filtered_sentences $FILTERED_SENTENCES_PATH  --filtered_tokens $FILTERED_TOKENS_PATH  --filtered_frequencies $FILTERED_FREQUENCIES_PATH

# Word2Vec training

EMBEDDING_SIZE="embedding300"

RANDOM_SIZE=100000000
EMBEDDING=300
ANNEAL_RATE=0.9
ANNEAL_EVERY=100000
PRINT_EVERY=10000
SENTENCES_PATH="$BASE_PATH-sentences-$DOC_FORMAT-$VOCAB_FORMAT.pickle"
TOKENS_PATH="$BASE_PATH-tokens-$DOC_FORMAT-$VOCAB_FORMAT.pickle"
FREQUENCIES_PATH="$BASE_PATH-frequencies-$DOC_FORMAT-$VOCAB_FORMAT.pickle"
W_IN_PATH="$BASE_PATH-win-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE.pickle"
W_OUT_PATH="$BASE_PATH-wout-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE.pickle"

python3 word2vec.py --random $RANDOM_SIZE --embedding $EMBEDDING --anneal_rate $ANNEAL_RATE --anneal $ANNEAL_EVERY --print $PRINT_EVERY --sentences_path $SENTENCES_PATH  --tokens_path $TOKENS_PATH  --frequencies_path $FREQUENCIES_PATH --w_in_path $W_IN_PATH --w_out_path $W_OUT_PATH


