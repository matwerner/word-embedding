BASE_PATH="../datasets/ptwiki-20170801"
VERBOSE="True"

# Preprocessing

DOC_FORMAT="doc1kk"

ARTICLES=1000000
CORPORA_PATH="$BASE_PATH-pages-articles-parsed.json"
SENTENCES_PATH="$BASE_PATH-$DOC_FORMAT-sentences.pickle"
TOKENS_PATH="$BASE_PATH-$DOC_FORMAT-tokens.pickle"
STOPWORDS="../datasets/stopwords.txt"

#python3 preprocessing.py preprocess --corpora $CORPORA_PATH --limit $ARTICLES --stopwords $STOPWORDS --sentences $SENTENCES_PATH --tokens $TOKENS_PATH --verbose $VERBOSE

VOCAB_FORMAT="vocab30k"

# Filter vocabulary

TOKENS=30000
FILTER_SENTENCES_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-sentences.pickle"
FILTER_TOKENS_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-tokens.pickle"

python3 preprocessing.py filter --limit $TOKENS --raw_sentences $SENTENCES_PATH --sentences $FILTER_SENTENCES_PATH --tokens $FILTER_TOKENS_PATH --verbose $VERBOSE

# Split sentences in training, validation and test sentences

TRAINING_SENTENCES_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-sentences-training.pickle"
VALIDATION_SENTENCES_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-sentences-validation.pickle"
TEST_SENTENCES_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-sentences-test.pickle"

python3 preprocessing.py split --sentences $FILTER_SENTENCES_PATH --percentage "70/20" --training $TRAINING_SENTENCES_PATH --validation $VALIDATION_SENTENCES_PATH --test $TEST_SENTENCES_PATH --verbose $VERBOSE
