BASE_PATH="../datasets/ptwiki-20170801"
DOC_FORMAT="doc1kk"
VOCAB_FORMAT="vocab30k"

TOKENS_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-tokens.pickle"
SENTENCES_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-sentences-training.pickle"

# Word2Vec training

RANDOM_SIZE=100000000
CONTEXT=5
NEGATIVE_SAMPLE=10
ANNEAL_RATE=0.9
ANNEAL_EVERY=100000
PRINT_EVERY=10000

# Embedding 1300

EMBEDDING_SIZE="embedding1300"
EMBEDDING=1300
W_IN_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-win.pickle"
W_OUT_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-wout.pickle"

python3 word2vec.py --random $RANDOM_SIZE --embedding $EMBEDDING --context $CONTEXT --negative_sample $NEGATIVE_SAMPLE --anneal_rate $ANNEAL_RATE --anneal $ANNEAL_EVERY --print $PRINT_EVERY --sentences_path $SENTENCES_PATH --tokens_path $TOKENS_PATH --w_in_path $W_IN_PATH --w_out_path $W_OUT_PATH

# Embedding 1500

EMBEDDING_SIZE="embedding1500"
EMBEDDING=1500
W_IN_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-win.pickle"
W_OUT_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-wout.pickle"

python3 word2vec.py --random $RANDOM_SIZE --embedding $EMBEDDING --context $CONTEXT --negative_sample $NEGATIVE_SAMPLE --anneal_rate $ANNEAL_RATE --anneal $ANNEAL_EVERY --print $PRINT_EVERY --sentences_path $SENTENCES_PATH --tokens_path $TOKENS_PATH --w_in_path $W_IN_PATH --w_out_path $W_OUT_PATH

# Embedding 900

EMBEDDING_SIZE="embedding900"
EMBEDDING=900

# Context 3

CONTEXT=3
CONTEXT_SIZE="context3"
W_IN_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-$CONTEXT_SIZE-win.pickle"
W_OUT_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-$CONTEXT_SIZE-wout.pickle"

#python3 word2vec.py --random $RANDOM_SIZE --embedding $EMBEDDING --context $CONTEXT --negative_sample $NEGATIVE_SAMPLE --anneal_rate $ANNEAL_RATE --anneal $ANNEAL_EVERY --print $PRINT_EVERY --sentences_path $SENTENCES_PATH --tokens_path $TOKENS_PATH --w_in_path $W_IN_PATH --w_out_path $W_OUT_PATH

# Context 7

CONTEXT=7
CONTEXT_SIZE="context7"
W_IN_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-$CONTEXT_SIZE-win.pickle"
W_OUT_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-$CONTEXT_SIZE-wout.pickle"

python3 word2vec.py --random $RANDOM_SIZE --embedding $EMBEDDING --context $CONTEXT --negative_sample $NEGATIVE_SAMPLE --anneal_rate $ANNEAL_RATE --anneal $ANNEAL_EVERY --print $PRINT_EVERY --sentences_path $SENTENCES_PATH --tokens_path $TOKENS_PATH --w_in_path $W_IN_PATH --w_out_path $W_OUT_PATH

# Context 10

CONTEXT=10
CONTEXT_SIZE="context10"
W_IN_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-$CONTEXT_SIZE-win.pickle"
W_OUT_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-$CONTEXT_SIZE-wout.pickle"

python3 word2vec.py --random $RANDOM_SIZE --embedding $EMBEDDING --context $CONTEXT --negative_sample $NEGATIVE_SAMPLE --anneal_rate $ANNEAL_RATE --anneal $ANNEAL_EVERY --print $PRINT_EVERY --sentences_path $SENTENCES_PATH --tokens_path $TOKENS_PATH --w_in_path $W_IN_PATH --w_out_path $W_OUT_PATH
