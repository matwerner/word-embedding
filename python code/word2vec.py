"""
Implementation of word2vec (skip-gram).

Tutorial: http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
"""

"""
    Load Packages
"""

import argparse
import numpy as np
import pickle
import util
import theano
import theano.tensor as T

# Fix seed
np.random.seed(100)

"""
    Word2vec Class - Skip-gram implementation
"""

class Word2Vec(object):
    
    def __init__(self,
                 corpus,
                 embedding_size=10,
                 unigram_table=None):
        
        self.corpus = corpus
        self.embedding_size = embedding_size
        self.unigram_table = unigram_table
        
        # Initializing network parameters
        self.W_in_values = np.asarray((np.random.rand(corpus.tokens_size(), embedding_size) - 0.5) / embedding_size,
                                      dtype=theano.config.floatX)

        self.W_out_values = np.asarray(np.zeros((corpus.tokens_size(), embedding_size)),
                                      dtype=theano.config.floatX)

        # Declaring theano parameters
        # Embedding variables
        self.W_in = theano.shared(
            value=self.W_in_values,
            name='W_in',
            borrow=True
        )

        self.W_out = theano.shared(
            value=self.W_out_values,
            name='W_out',
            borrow=True
        )
        
        # Get training model
        self.train_model = self.__train_model()

    def __train_model(self):
        # Input variables
        target_words = T.ivector('target_words')
        context_words = T.ivector('context_words')
        in_corpus = T.ivector('in_corpus')
        learning_rate = T.scalar('learning_rate')
        
        # Prepare word embeddings
        target_embedding = self.W_in[target_words]
        context_embedding = self.W_out[context_words]
        
        # Compute cost
        positive_cost = in_corpus * T.log(T.nnet.sigmoid(T.sum(target_embedding * context_embedding, axis=1)))
        negative_cost = (1 - in_corpus) * T.log(T.nnet.sigmoid(-T.sum(target_embedding * context_embedding, axis=1)))        
        cost = -T.sum(positive_cost + negative_cost)
        
        # Compute gradient        
        grad_in, grad_out = T.grad(cost, [target_embedding, context_embedding])
        
        # Zip updates
        updates = [(self.W_in, T.inc_subtensor(target_embedding, - learning_rate * grad_in)),
                   (self.W_out, T.inc_subtensor(context_embedding, - learning_rate * grad_out))]
        
        # Create theano training function
        train_model = theano.function(
            inputs=[target_words,
                    context_words,
                    in_corpus,
                    learning_rate],
            outputs=cost,
            updates=updates,
            profile=True
        )
        
        return train_model

    def train(self,
              random_size=10**8,
              window_size=5,
              negative_sample_size=5,
              learning_rate=0.3,              
              batch_size=100,
              anneal_rate=0.9,
              anneal_every=100000,
              print_every=5000):
        
        print('Start Training')

        # Batch variables
        center_words = []
        contexts = []
        in_corpus = []
        
        batch_cost = 0
        for it, (center_word, context) in enumerate(self.corpus.random_contexts(random_size, window_size)):
            # Define constants
            context_size = len(context)
            total_negative_sample_size = context_size * negative_sample_size
            center_word_size = context_size + total_negative_sample_size
            
            # Generate negative sample
            negative_samples = self.unigram_table.sample(total_negative_sample_size)
            
            # Increment batch
            center_words +=  center_word_size * [center_word]
            contexts += (context + negative_samples.tolist())
            in_corpus += (context_size * [1] + total_negative_sample_size * [0])
            
            # Gathered contexts until batch size
            if (it + 1) % batch_size != 0: 
                continue
            
            # Train for many contexts
            batch_cost += self.train_model(center_words,
                                           contexts,
                                           in_corpus,
                                           learning_rate)

            # Update learning rate
            if (it + 1) % anneal_every == 0:
                learning_rate *= anneal_rate

            # Print temp results
            if (it + 1) % print_every == 0:
                print('Iteration:{}, Batch Cost {}'.format(it + 1, batch_cost/print_every))
                batch_cost = 0
            
            # Empty batch
            center_words = []
            contexts = []
            in_corpus = []
        self.train_model.profile.summary()
        return batch_cost
    
    def save(self, W_in_path, W_out_path):
        with open(W_in_path, 'wb') as fp:
            pickle.dump(self.W_in_values, fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(W_out_path, 'wb') as fp:
            pickle.dump(self.W_out_values, fp, protocol=pickle.HIGHEST_PROTOCOL)

"""
    Experiments
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Word Embedding using Word2Vec')
    parser.add_argument('--sampling', dest='sampling_rate',
                        type=float, default=1e-3,
                        help='Subsampling of frequenty words')
    parser.add_argument('--learning_rate', dest='learning_rate',
                        type=float, default=0.025,
                        help='Starting learning rate of the algorithm')                        
    parser.add_argument('--embedding', dest='embedding_size',
                        type=int, default=300,
                        help='Embedding dimensionality')
    parser.add_argument('--random', dest='random_size',
                        type=int, default=10**8,
                        help='Number of random contexts to train the word embedding')
    parser.add_argument('--context', dest='context_size',
                        type=int, default=5,
                        help='Window size')
    parser.add_argument('--negative_sample', dest='negative_sample_size',
                        type=int, default=10,
                        help='Number of counter-examples')
    parser.add_argument('--batch', dest='batch_size',
                        type=int, default=100,
                        help='Number of training samples before update weights')
    parser.add_argument('--anneal_rate', dest='anneal_rate',
                        type=float, default=0.9,
                        help='Rate for update Learning Rate')
    parser.add_argument('--anneal', dest='anneal_every',
                        type=int, default=100000,
                        help='Frequency of update Learning Rate')
    parser.add_argument('--print', dest='print_every',
                        type=int, default=5000,
                        help='Frequency of print temporary results')
    parser.add_argument('--sentences_path', dest='sentences_path', type=str,
                        help='Sentences dataset path')
    parser.add_argument('--tokens_path', dest='tokens_path', type=str,
                        help='Tokens dataset path')
    parser.add_argument('--w_in_path', dest='w_in_path', type=str,
                        help='Word Embeddings IN path')
    parser.add_argument('--w_out_path', dest='w_out_path', type=str,
                        help='Word Embeddings OUT path')
    
    args = parser.parse_args()

    """
        Load corpus
    """

    corpus = util.Corpus(args.sentences_path, args.tokens_path, args.sampling_rate)

    """
        Process unigram table
    """

    unigram_table = util.UnigramTable(corpus.frequencies())

    """
        Run
    """

    word2vec = Word2Vec(corpus, args.embedding_size, unigram_table)
    word2vec.train(random_size=args.random_size,
                   window_size=args.context_size,
                   negative_sample_size=args.negative_sample_size,
                   batch_size=args.batch_size,
                   learning_rate=args.learning_rate,
                   anneal_rate=args.anneal_rate,
                   anneal_every=args.anneal_every * args.batch_size,
                   print_every=args.print_every * args.batch_size)

    word2vec.save(args.w_in_path, args.w_out_path)