"""
Datasetken: https://dumps.wikimedia.org/backup-index.html

Before Running the parser below, please run WikipediaExtractor script:

python2 WikipediaExtractor.py --json --bytes 10G

Download: https://github.com/attardi/wikiextractor

Source: https://www.mediawiki.org/wiki/Alternative_parsers
"""

"""
    Load Packages
"""
from collections import Counter

import argparse
import io
import json
import math
import os.path
import pickle
import re
import string

VERBOSE_EVERY_PERC = 0.1

"""
    Convert Corpus to sentences
"""

numbers = '0123456789'
punctuation = re.sub('[-,:=!?\.]', '', string.punctuation)
translator = str.maketrans('', '', punctuation + numbers) 

def get_sentences(corpus):
    # Remove punctuation and numbers
    preprocessed = re.sub('[-,–:=]', ' ', corpus)
    preprocessed = preprocessed.translate(translator)
    
    # Split text into sentences
    sentences = re.split('[!?\.\\n]', preprocessed)
    
    # Remove spaces from start and end of sentence
    sentences = [sentence.strip()
                 for sentence in sentences]
    
    # First caracter of each sentence to lower
    # Each sentence must have two or more words
    sentences = [sentence[:1].lower() + sentence[1:]
                 for sentence in sentences
                 if sentence and ' '  in sentence]
    
    # Split sentence in tokens
    sentences = [' '.join(re.split('\s+', sentence))
                 for sentence in sentences]
    
    return sentences

"""
    Learn Phrases
"""

def learn_phrases(sentences, min_count=5, threshold=10, token_limit=10e7,
                  delimiter='_', token_delimiter=' ', stopwords=frozenset(),
                  verbose=False):
    if verbose:
        print("Starting Learning Phrases")

    # All variables
    idx = 0
    tokens = {}
    token_freq = Counter()
    phrase_freq = Counter()
    word_count = 0
    unique_count = 0
    print_every = int(VERBOSE_EVERY_PERC * (len(sentences)-1))

    phrase_format = '%s' + delimiter + '%s'
    
    # Get counts
    for i, sentence in enumerate(sentences):
        if verbose and i % print_every == 0:
            perc = math.ceil(100 * i/len(sentences))
            print("Learned possible phrases from %f%% of the sentences" % (perc,))

        previous_word = None
        for word in sentence.split(token_delimiter):
            if unique_count > token_limit:
                break
                
            # Update token frequency
            token_freq[word] += 1

            # Update tokens
            if word not in tokens:
                tokens[word] = idx
                unique_count += 1
                idx += 1

            # Update phrases
            if previous_word:
                # Check for stopwords
                if previous_word.lower() in stopwords or word.lower() in stopwords:
                    previous_word = word
                    continue
                phrase = phrase_format % (previous_word, word)
                phrase_freq[phrase] += 1
                unique_count += 1

            # Next
            previous_word = word
            word_count += 1
        if unique_count > token_limit:
            if verbose:
                print('Limit reached')
            break
                
    # Find valid phrases
    print_every = int(VERBOSE_EVERY_PERC * (len(phrase_freq)-1))
    valid_phrases = Counter()
    for phrase, count in phrase_freq.items():
        if verbose and i % print_every == 0:
            perc = math.ceil(100 * i/len(phrase_freq))
            print("Checked valid phrases from %f%% of the possible phrases" % (perc,))

        word_a, word_b = phrase.split(delimiter)
        
        # Get counts
        count_a, count_b = token_freq[word_a], token_freq[word_b]
        
        # Calculate score
        score = word_count * (count - min_count)/ (count_a * count_b)
        
        # Append if valid
        if score > threshold:
            valid_phrases[phrase] = count

    if verbose:
        print("Total phrases:\t%d" % (len(phrase_freq),))
        print("Valid phrases:\t%d" % (len(valid_phrases),))
        print("Learning Phrases completed\n")
            
    return valid_phrases

"""
    Replace all tokens pairs by their respectives phrases
"""

def replace_tokens_to_phrases(sentences, phrases, delimiter='_', token_delimiter=' ', verbose=False):
    if verbose:
        print("Starting Replace tokens to Phrases")

    phrase_format = '%s' + delimiter + '%s'
    print_every = int(VERBOSE_EVERY_PERC * (len(sentences)-1))
    
    # Update sentences
    for i in range(len(sentences)):
        if verbose and i % print_every == 0:
            perc = math.ceil(100 * i/len(sentences))
            print("Replaced tokens from %f%% of the sentences" % (perc,))

        # Get sentence
        sentence = sentences[i].split(token_delimiter)
        
        new_sentence = ''
        concat_phrase = None
        previous_word = sentence[0]
        for word in sentence[1:]:
            phrase = phrase_format % (previous_word, word)
            if phrase in phrases:
                concat_phrase = phrase if not concat_phrase else phrase_format % (concat_phrase, word)
            else:
                new_sentence += ((concat_phrase if concat_phrase else previous_word) + token_delimiter)
                concat_phrase = None
            previous_word = word
        new_sentence += (concat_phrase if concat_phrase else previous_word)
        sentences[i] = new_sentence

    if verbose:
        print("Replace tokens to Phrases completed")

    return sentences

"""
    Convert Corpora to Sentences
"""

def corpora_to_sentences(corpora, stopwords=[], token_delimiter=' ', verbose=False):
    if verbose:
        print("Starting Corpora to Sentences")

    # All variables
    sentences = []
    idx = 0
    tokens = {}
    token_freq = Counter()
    word_count = 0
    print_every = int(VERBOSE_EVERY_PERC * (len(corpora)-1))    

    # Preprocess all corpora
    for i, corpus in enumerate(corpora):
        if verbose and i % print_every == 0:
            perc = math.ceil(100 * i/len(corpora))
            print("Preprocessed %f%% of the corpora" % (perc,))
        sentences += get_sentences(corpus)
    
    # Load stopwords
    exception = set(['de', 'do', 'dos', 'da', 'das', 'ser'])
    stopwords = set(stopwords) - exception
    stopwords = frozenset(stopwords)
    
    # Get phrases
    phrases_freq = learn_phrases(sentences, stopwords=stopwords, verbose=verbose)
    
    # Update sentences with new phrases
    sentences = replace_tokens_to_phrases(sentences, phrases_freq, verbose=verbose)
    
    # Get counters
    for sentence in sentences:
        for word in sentence.split(token_delimiter):                
            if word not in tokens:
                tokens[word] = idx
                token_freq[word] = 1
                idx += 1
            else:
                token_freq[word] += 1
            word_count += 1

    if verbose:
        print("Sentences:\t%d" % (len(sentences),))
        print("Tokens:\t\t%d" % (len(token_freq),))
        print("Word Count:\t%d" % (word_count,))
        print("Corpora to Sentences completed")
 
    return sentences, tokens, token_freq, phrases_freq, word_count

"""
    Filter vocabulary by most common words
"""

def filter_vocabulary(sentences, tokens, token_freq, max_tokens, token_delimiter=' '):
    # Get most frequent words
    token_freq = Counter(dict(token_freq.most_common(max_tokens)))
    
    # Remove all other words from indices
    tokens = {key:i for i, key in enumerate(token_freq.keys())}
    
    # Apply remmap in all data structures
    sentences = [token_delimiter.join([token
                                       for token in sentence.split(token_delimiter)
                                       if token in tokens])
                 for sentence in sentences]
    word_count = sum(token_freq.values())
    
    return sentences, tokens, token_freq, word_count


"""
    Run experiments
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process wikipedia dump articles')
    parser.add_argument('--n_articles', dest='max_articles', metavar='N', type=int,
                        help='number maximum of articles to be processed')
    parser.add_argument('--n_tokens', dest='max_tokens', metavar='N', type=int,
                        help='number maximum of tokens to be used')    
    parser.add_argument('--wiki_dump', dest='dump_path', metavar='N', type=str,
                        help='path to wikipedia dump articles')
    parser.add_argument('--full_sentences', dest='full_sentences_path', metavar='N', type=str,
                        help='path to store all sentences of processed articles')
    parser.add_argument('--full_tokens', dest='full_tokens_path', metavar='N', type=str,
                        help='path to store all tokens of processed articles')
    parser.add_argument('--full_frequencies', dest='full_frequencies_path', metavar='N', type=str,
                        help='path to store all frequencies of processed articles')
    parser.add_argument('--filtered_sentences', dest='filtered_sentences_path', metavar='N', type=str,
                        help='path to store all sentences of processed and filtered articles')
    parser.add_argument('--filtered_tokens', dest='filtered_tokens_path', metavar='N', type=str,
                        help='path to store all tokens of processed and filtered articles')
    parser.add_argument('--filtered_frequencies', dest='filtered_frequencies_path', metavar='N', type=str,
                        help='path to store all frequencies of processed and filtered articles')
    parser.add_argument('--stopwords', dest='stopwords_path', metavar='N', type=str,
                        help='path stopwords to be ignored while learning phrases')
    parser.add_argument('--verbose', dest='verbose', metavar='N', type=bool,
                        help='Print Progress')                        
    
    args = parser.parse_args()

    """
        Data structures to be used
    """

    sentences = []
    tokens = {}
    token_freq = Counter()
    phrase_freq = Counter()
    word_count = 0

    """
        Convert all articles to sentences
    """

    # Check if already converted
    if os.path.isfile(args.full_sentences_path) and not os.path.isfile(args.filtered_sentences_path):
        if args.verbose:
            print('Loading full vocabulary sentences')

        with open(args.full_sentences_path, 'rb') as fp:
            sentences = pickle.load(fp)
        with open(args.full_tokens_path, 'rb') as fp:
            tokens = pickle.load(fp) 
        with open(args.full_frequencies_path, 'rb') as fp:
            token_freq = pickle.load(fp) 
        word_count = sum(token_freq.values())

        if args.verbose:
            print('Loading completed!\n')

    elif not os.path.isfile(args.full_sentences_path):
        if args.verbose:
            print('Start processing articles')

        # Load data
        data_reader = io.open(args.dump_path, mode="r", encoding="utf-8")

        # Go to beginning
        data_reader.seek(0)

        # Parse all text from json
        articles = [json.loads(line)['text']
                    for line in data_reader]

        data_reader.close()

        # Temporary - Remove once word embedding algorithms are ready
        articles = articles[:args.max_articles]

        # Run
        stopwords = [] if not args.stopwords_path else [line.strip() for line in open(args.stopwords_path)]
        sentences, tokens, token_freq, phrases_freq, word_count = corpora_to_sentences(articles, stopwords, verbose=args.verbose)
        
        # Delete articles - memory restraint
        del articles

        # Save indexes
        with open(args.full_sentences_path, 'wb') as fp:
            pickle.dump(sentences, fp, protocol=pickle.HIGHEST_PROTOCOL) 
        with open(args.full_tokens_path, 'wb') as fp:
            pickle.dump(tokens, fp, protocol=pickle.HIGHEST_PROTOCOL) 
        with open(args.full_frequencies_path, 'wb') as fp:
            pickle.dump(token_freq, fp, protocol=pickle.HIGHEST_PROTOCOL)
        
        if args.verbose:
            print('Processing completed\n')
            print(phrases_freq.most_common(100))

    """
        Limit Tokens to be used
    """

    if not os.path.isfile(args.filtered_sentences_path):
        if args.verbose:
            print('Filtering vocabulary from sentences')

        # Run
        sentences, tokens, token_freq, word_count = filter_vocabulary(sentences,
                                                                      tokens,
                                                                      token_freq,
                                                                      args.max_tokens)
        
        # Save indexes
        with open(args.filtered_sentences_path, 'wb') as fp:
            pickle.dump(sentences, fp, protocol=pickle.HIGHEST_PROTOCOL) 
        with open(args.filtered_tokens_path, 'wb') as fp:
            pickle.dump(tokens, fp, protocol=pickle.HIGHEST_PROTOCOL) 
        with open(args.filtered_frequencies_path, 'wb') as fp:
            pickle.dump(token_freq, fp, protocol=pickle.HIGHEST_PROTOCOL)
            
        if args.verbose:
            print('Filtering completed\n')
