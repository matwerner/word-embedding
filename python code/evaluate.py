import argparse
import numpy as np
import pickle
import util

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Word Embedding Test')
    parser.add_argument('--training_path', dest='training_path', type=str,
                        help='Training sentences dataset path')
    parser.add_argument('--validation_path', dest='validation_path', type=str,
                        help='Validation sentences dataset path')
    parser.add_argument('--tokens_path', dest='tokens_path', type=str,
                        help='Tokens dataset path')
    parser.add_argument('--w_in_path', dest='w_in_path', type=str,
                        help='Word Embeddings IN path')
    parser.add_argument('--w_out_path', dest='w_out_path', type=str,
                        help='Word Embeddings OUT path')
    parser.add_argument('--test', dest='test_size',
                        type=int, default=10**6,
                        help='Number of test instances')
    parser.add_argument('--context', dest='context_size',
                        type=int, default=5,
                        help='Window size')
    parser.add_argument('--negative_sample', dest='negative_sample_size',
                        type=int, default=10,
                        help='Number of counter-examples')
    args = parser.parse_args()

    """
        load corpus
    """

    validation_corpus = util.Corpus(args.validation_path, args.tokens_path, 10e-3)

    """
        Process unigram table
    """

    training_corpus = util.Corpus(args.training_path, args.tokens_path, 10e-3)
    unigram_table = util.UnigramTable(training_corpus.frequencies())

    
    """
        Load embeddings
    """

    with open(args.w_in_path, 'rb') as fp:
        w_in = pickle.load(fp)
    with open(args.w_out_path, 'rb') as fp:
        w_out = pickle.load(fp)
    embeddings = w_in + w_out

    """
        Create test and evaluate embeddings
    """

    test = util.FakeTest()
    test.create(validation_corpus, unigram_table, args.context_size,  args.negative_sample_size, args.test_size)
    correct, incorrect = test.evaluate(embeddings)

    """
        Print results
    """

    n_correct, n_incorrect = len(correct), len(incorrect)
    n_total = n_correct + n_incorrect
    print("Embedding size:\t%d" % embeddings.shape[1])
    print("Context size:\t%d" % args.context_size)
    print("Negative sample:\t%d" % args.negative_sample_size)
    print("Result = %f%% (%d of %d)\n" % (100 * n_correct/n_total, n_correct, n_total))
