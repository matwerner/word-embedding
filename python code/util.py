"""
    Load Packages
"""

from collections import Counter

import numpy as np
import pickle

# Fix seed
np.random.seed(100)

"""
Corpus Class
Corpus dataset must be preprocessed by preprocessing.ipynb before being load
"""

class Corpus(object):

    def __init__(self, sentences_path, tokens_path, sampling_rate, delimiter=' '):
        # Sentence are store as string not as vectors
        self.__delimiter = delimiter
        
        # Rate for decrease words
        self.sampling_rate = sampling_rate
        
        # Load Corpus
        with open(tokens_path, 'rb') as fp:
            self.__tokens = pickle.load(fp)
        with open(sentences_path, 'rb') as fp:
            self.__sentences = pickle.load(fp)

        # Compute frequencies
        frequencies = Counter()
        for sentence in self.__sentences:
            for token in sentence.split(delimiter):
                frequencies[token] += 1
        self.__frequencies = frequencies
    
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
        
        self.__word_size = sum(self.__frequencies.values())
        return self.__word_size

    def frequencies(self):
        return {self.__tokens[token]:count
                for token, count in self.__frequencies.items()}
    
    def tokens(self):
        return self.__tokens.copy()
    
    def rejection_probability(self):
        if hasattr(self, '__rejection_probability') and self.__rejection_probability:
            return self.__reject_prob

        n_words = self.words_size()
        n_tokens = self.tokens_size()
        rejection_probability = {}
        for key, count in self.__frequencies.items():
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
                    if 0 > prob or np.random.random() > prob:
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
            sentence_idx = np.random.randint(0, len(sentences) - 1)
            sentence = sentences[sentence_idx]
            indexes = self.__get_sentence_indexes(sentence)

            # Get random center word
            center_idx = np.random.randint(0, len(indexes) - 1)
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
        return self.__delimiter.join(tokens)
    
    def __get_sentence_tokens(self, sentence):
        return [word for word in sentence.split(self.__delimiter)]
    
    def __get_sentence_indexes(self, sentence):
        return [self.__tokens[word] for word in sentence.split(self.__delimiter)]

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
    Fake Test Class
"""

class FakeTest(object):

    def __init__(self):
        # We should not have the same seed as corpus contexts
        np.random.seed(200)
        self.instances = None

    def create(self, corpus, unigram_table, 
               window_size, negative_sample_size, test_size):
        self.instances = []
        for center_word, context in corpus.random_contexts(test_size, window_size):
            if len(context) < 2:
                continue

            # Get words not in context
            negative_words = tuple(unigram_table.sample(negative_sample_size))

            # Get random context word
            idx = np.random.randint(0, len(context))

            # Append to fake test
            instance = (center_word, context[idx]) + negative_words
            self.instances.append(instance)

    def save(self, output_path):
        with open(output_path, 'wb') as fp:
            pickle.dump(self.instances, fp, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, test_path):
        with open(test_path, 'rb') as fp:
            self.instances = pickle.load(fp)

    def evaluate(self, embeddings):
        correct = []
        incorrect = []
        features = embeddings.shape[1]
        for test_words in self.instances:
            center_embedding = embeddings[test_words[0]]
            test_embeddings = [embeddings[word] for word in test_words[1:]]

            prob = 1./(1. + np.exp(-np.sum(center_embedding * test_embeddings, axis=1)))
            result = np.argmax(prob)

            if result == 0:
                correct.append((test_words[0], test_words[1]))
            else:
                incorrect.append((test_words[0], test_words[1], test_words[result + 1]))

        return correct, incorrect
