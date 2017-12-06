BASE_PATH="../datasets/ptwiki-20170801"
DOC_FORMAT="doc1kk"
VOCAB_FORMAT="vocab30k"

TOKENS_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-tokens.pickle"
TRAINING_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-sentences-training.pickle"
VALIDATION_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-sentences-validation.pickle"
RESULTS_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-results.txt"

# Test

TEST_SIZE=1000000
CONTEXT=5
NEGATIVE_SAMPLE=10

# Embedding 1300

EMBEDDING_SIZE="embedding1300"
W_IN_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-win.pickle"
W_OUT_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-wout.pickle"

python3 evaluate.py --test $TEST_SIZE --context $CONTEXT --negative_sample $NEGATIVE_SAMPLE --training_path $TRAINING_PATH --validation_path $VALIDATION_PATH --tokens_path $TOKENS_PATH --w_in_path $W_IN_PATH --w_out_path $W_OUT_PATH | tee -a $RESULTS_PATH

# Embedding 1500

EMBEDDING_SIZE="embedding1500"
W_IN_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-win.pickle"
W_OUT_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-wout.pickle"

python3 evaluate.py --test $TEST_SIZE --context $CONTEXT --negative_sample $NEGATIVE_SAMPLE --training_path $TRAINING_PATH --validation_path $VALIDATION_PATH --tokens_path $TOKENS_PATH --w_in_path $W_IN_PATH --w_out_path $W_OUT_PATH | tee -a $RESULTS_PATH

# Embedding 900

EMBEDDING_SIZE="embedding900"

# CONTEXT 3

CONTEXT=3
CONTEXT_SIZE="context3"
W_IN_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-$CONTEXT_SIZE-win.pickle"
W_OUT_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-$CONTEXT_SIZE-wout.pickle"

#python3 evaluate.py --test $TEST_SIZE --context $CONTEXT --negative_sample $NEGATIVE_SAMPLE --training_path $TRAINING_PATH --validation_path $VALIDATION_PATH --tokens_path $TOKENS_PATH --w_in_path $W_IN_PATH --w_out_path $W_OUT_PATH | tee -a $RESULTS_PATH

# CONTEXT 7

CONTEXT=7
CONTEXT_SIZE="context7"
W_IN_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-$CONTEXT_SIZE-win.pickle"
W_OUT_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-$CONTEXT_SIZE-wout.pickle"

#python3 evaluate.py --test $TEST_SIZE --context $CONTEXT --negative_sample $NEGATIVE_SAMPLE --training_path $TRAINING_PATH --validation_path $VALIDATION_PATH --tokens_path $TOKENS_PATH --w_in_path $W_IN_PATH --w_out_path $W_OUT_PATH | tee -a $RESULTS_PATH

# Embedding 700

CONTEXT=10
CONTEXT_SIZE="context10"
W_IN_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-$CONTEXT_SIZE-win.pickle"
W_OUT_PATH="$BASE_PATH-$DOC_FORMAT-$VOCAB_FORMAT-$EMBEDDING_SIZE-$CONTEXT_SIZE-wout.pickle"

#python3 evaluate.py --test $TEST_SIZE --context $CONTEXT --negative_sample $NEGATIVE_SAMPLE --training_path $TRAINING_PATH --validation_path $VALIDATION_PATH --tokens_path $TOKENS_PATH --w_in_path $W_IN_PATH --w_out_path $W_OUT_PATH | tee -a $RESULTS_PATH
