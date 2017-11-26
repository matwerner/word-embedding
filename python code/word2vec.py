"""
Implementation of word2vec (skip-gram).

Tutorial: http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
"""

"""
    Load Packages
"""

import argparse
import numpy as np
import _pickle as cPickle
import random
import theano
import theano.tensor as T

"""
    Corpus Class
    Corpus dataset must be preprocessed by preprocessing.ipynb before being load
"""

class Corpus(object):
    
    def __init__(self, corpus_path, sampling_rate, token_delimiter=' '):
        # Sentence are store as string not as vectors
        self.__token_delimiter = token_delimiter
        
        # Rate for decrease words
        self.sampling_rate = sampling_rate
        
        # Load Corpus
        with open(corpus_path, 'rb') as fp:
            self.__tokens = cPickle.load(fp) 
            self.__token_freq = cPickle.load(fp)
            self.__sentences = cPickle.load(fp)
    
    def sentences_size(self):
        if hasattr(self, "__sentences_size") and self.__sentences_size:
            return self.__sentences_size
        
        self.__sentences_size = len(self.__sentences)
        return self.__sentences_size
    
    def tokens_size(self):
        if hasattr(self, "__tokens_size") and self.__tokens_size:
            return self.__tokens_size
        
        self.__tokens_size = len(self.__tokens)
        return self.__tokens_size
    
    def words_size(self):
        if hasattr(self, "__words_size") and self.__words_size:
            return self.__words_size
        
        self.__word_size = sum(self.__token_freq.values())
        return self.__word_size

    def frequencies(self):
        return {self.__tokens[token]:count
                for token, count in self.__token_freq.items()}
    
    def tokens(self):
        return self.__tokens.copy()
    
    def rejection_probability(self):
        if hasattr(self, '__rejection_probability') and self.__rejection_probability:
            return self.__reject_prob

        n_words = self.words_size()
        n_tokens = self.tokens_size()
        rejection_probability = {}
        for key, count in self.__token_freq.items():
            density = count/(1.0 * n_words)            
            
            # Calculate rejection probability
            rejection_probability[key] = 1 - (np.sqrt(density/self.sampling_rate) + 1) * (self.sampling_rate/density)

        self.__rejection_probability = rejection_probability
        return self.__rejection_probability

    def sentences(self):
        if hasattr(self, "__process_sentences") and self.__process_sentences:
            return self.__sentences
        self.__process_sentences = True
        
        rejection_probability = self.rejection_probability()
        
        j = 0
        for i in range(self.sentences_size()):
            tokens = []
            for word in self.__get_sentence_tokens(self.__sentences[i]):
                if not word:
                    continue
                try:
                    prob = rejection_probability[word]
                    if 0 > prob or random.random() > prob:
                        tokens += [word]
                except:
                    raise ValueError("Line %d: %s" % (i, self.__sentences[i]))
            
            if len(tokens) < 2:
                continue
            
            self.__sentences[j] = self.__set_sentence_tokens(tokens)
            j+=1
        
        self.__sentences_size = j
        self.__sentences = self.__sentences[:j]
        return self.__sentences

    def contexts(self, C=5):
        for sentence in self.sentences():
            indexes = self.__get_sentence_indexes(sentence)
            for center_idx, center_word in enumerate(indexes):
                # Get current context
                context = self.__get_context(indexes, center_idx, center_word, C)

                # Return current context
                yield center_word, context
                
    def random_contexts(self, size, C=5):
        sentences = self.sentences()
        for _ in range(size):
            # Get random sentence
            sentence_idx = random.randint(0, len(sentences) - 1)
            sentence = sentences[sentence_idx]
            indexes = self.__get_sentence_indexes(raw_sentence)

            # Get random center word
            center_idx = random.randint(0, len(indexes) - 1)
            center_word = indexes[center_idx]

            # Get current context
            context = self.__get_context(indexes, center_idx, center_word, C)
            
            # Return current context
            yield center_word, context
    
    def __get_context(self, sentence, center_idx, center_word, C=5):
        # Get previous words
        context = sentence[max(0, center_idx - C):center_idx]

        # Get future words
        if center_idx + 1 < len(sentence):
            context += sentence[center_idx+1:min(len(sentence), center_idx + C + 1)]

        # Remove duplicate center word
        context = [word for word in context if word is not center_word]
        
        return context
    
    def __set_sentence_tokens(self, tokens):
        return self.__token_delimiter.join(tokens)
    
    def __get_sentence_tokens(self, sentence):
        return [word for word in sentence.split(self.__token_delimiter)]
    
    def __get_sentence_indexes(self, sentence):
        return [self.__tokens[word] for word in sentence.split(self.__token_delimiter)]

"""
    Unigram table Class
"""

class UnigramTable(object):
    
    def __init__(self, counts):
        power = 0.75
        
        # Calculate distribution
        word_distribution = np.array([np.power(count, power) for count in counts.values()])
        
        # Normalize
        word_distribution /= np.sum(word_distribution)
        
        
        # table_size should be big enough so that the minimum probability for a word * table_size >= 1.
        # Also,  table_size must be hardcoded as a counter-measure for the case of the minimum probability
        # be a extremely low value, what would burst our memory
        table_size = int(1e8)
        table = np.zeros(table_size, dtype=np.int32)
        
        # Cumulative probability
        cum_probability = 0
        
        i = 0
        for word, count in counts.items():
            cum_probability += word_distribution[word]
            # fill the table until reach the cumulative probability
            while i < table_size and i / table_size < cum_probability:
                table[i] = word
                i += 1

        self.__table = table         
        self.__table_size = table_size

    def sample(self, k):        
        indices = np.random.randint(low=0, high=self.__table_size, size=k)
        return self.__table[indices]

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
              window_size=5,
              negative_sample_size=5,
              learning_rate=0.3,              
              batch_size=100,
              anneal_every=100000,
              print_every=5000):
        
        print('Start Training')

        # Batch variables
        center_words = []
        contexts = []
        in_corpus = []
        
        batch_cost = 0
        for it, (center_word, context) in enumerate(self.corpus.contexts(window_size)):
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
                learning_rate *= 0.5

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
    
    def save(self, output_path):
        with open(output_path, 'wb') as fp:
            cPickle.dump(self.W_in_values.shape, fp)
            cPickle.dump(self.W_in_values, fp)
            cPickle.dump(self.W_out_values, fp)

"""
    Experiments
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Word Embedding using Word2Vec')
    parser.add_argument('--sampling', dest='sampling_rate', metavar='N',
                        type=float, default=1e-3,
                        help='Subsampling of frequenty words')
    parser.add_argument('--learning_rate', dest='learning_rate', metavar='N',
                        type=float, default=0.025,
                        help='Starting learning rate of the algorithm')                        
    parser.add_argument('--embedding', dest='embedding_size', metavar='N',
                        type=int, default=300,
                        help='Embedding dimensionality')
    parser.add_argument('--context', dest='context_size', metavar='N',
                        type=int, default=5,
                        help='Window size')
    parser.add_argument('--negative_sample', dest='negative_sample_size', metavar='N',
                        type=int, default=10,
                        help='Number of counter-examples')
    parser.add_argument('--batch', dest='batch_size', metavar='N',
                        type=int, default=100,
                        help='Number of training samples before update weights')
    parser.add_argument('--annel', dest='anneal_every', metavar='N',
                        type=int, default=100000,
                        help='Update Learning Rate')
    parser.add_argument('--print', dest='print_every', metavar='N',
                        type=int, default=5000,
                        help='Frequency of print temporary results')
    parser.add_argument('--corpus_path', dest='corpus_path', metavar='N', type=str,
                        help='Corpus dataset path')
    parser.add_argument('--embedding_path', dest='word_embedding_path', metavar='N', type=str,
                        help='Word Embeddings path')
    
    args = parser.parse_args()

    """
        Load corpus
    """

    corpus = Corpus(args.corpus_path, args.sampling_rate)

    """
        Process unigram table
    """

    unigram_table = UnigramTable(corpus.frequencies())

    """
        Run
    """

    word2vec = Word2Vec(corpus, args.embedding_size, unigram_table)
    word2vec.train(window_size=args.context_size,
                   negative_sample_size=args.negative_sample_size,
                   batch_size=args.batch_size,
                   learning_rate=args.learning_rate,
                   anneal_every=args.anneal_every * args.batch_size,
                   print_every=args.print_every * args.batch_size)

    word2vec.save(args.word_embedding_path)