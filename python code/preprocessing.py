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
                  delimiter=' ', phrase_delimiter='_', stopwords=frozenset(),
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

    phrase_format = '%s' + phrase_delimiter + '%s'
    
    # Get counts
    for i, sentence in enumerate(sentences):
        if verbose and i % print_every == 0:
            perc = math.ceil(100 * i/len(sentences))
            print("Learned possible phrases from %f%% of the sentences" % (perc,))

        previous_word = None
        for word in sentence.split(delimiter):
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

        word_a, word_b = phrase.split(phrase_delimiter)
        
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

def replace_tokens_to_phrases(sentences, phrases, delimiter=' ', phrase_delimiter='_', verbose=False):
    if verbose:
        print("Starting Replace tokens to Phrases")

    phrase_format = '%s' + phrase_delimiter + '%s'
    print_every = int(VERBOSE_EVERY_PERC * (len(sentences)-1))
    
    # Update sentences
    for i in range(len(sentences)):
        if verbose and i % print_every == 0:
            perc = math.ceil(100 * i/len(sentences))
            print("Replaced tokens from %f%% of the sentences" % (perc,))

        # Get sentence
        sentence = sentences[i].split(delimiter)
        
        new_sentence = ''
        concat_phrase = None
        previous_word = sentence[0]
        for word in sentence[1:]:
            phrase = phrase_format % (previous_word, word)
            if phrase in phrases:
                concat_phrase = phrase if not concat_phrase else phrase_format % (concat_phrase, word)
            else:
                new_sentence += ((concat_phrase if concat_phrase else previous_word) + delimiter)
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

def corpora_to_sentences(corpora, stopwords=[], delimiter=' ', verbose=False):
    if verbose:
        print("Starting Corpora to Sentences")

    # All variables
    sentences = []
    idx = 0
    tokens = {}
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
    phrases = learn_phrases(sentences, stopwords=stopwords, verbose=verbose)
    
    # Update sentences with new phrases
    sentences = replace_tokens_to_phrases(sentences, phrases, verbose=verbose)
    
    # Get tokens
    for sentence in sentences:
        for word in sentence.split(delimiter):                
            if word not in tokens:
                tokens[word] = idx
                idx += 1
            word_count += 1

    if verbose:
        print("Sentences:\t%d" % (len(sentences),))
        print("Tokens:\t\t%d" % (len(tokens),))
        print("Word Count:\t%d" % (word_count,))
        print("Corpora to Sentences completed")
 
    return sentences, tokens

"""
    Filter vocabulary by most common words
"""

def filter_vocabulary(sentences, limit, delimiter=' ', verbose=False):
    if verbose:
        print("Starting filter vocabulary")
        
    # Get counters
    frequencies = Counter()
    for sentence in sentences:
        for word in sentence.split(delimiter):
            frequencies[word] += 1
    
    # Get most frequent words
    frequencies = Counter(dict(frequencies.most_common(limit)))
    
    # Remove all other words from indices
    tokens = {key:i for i, key in enumerate(frequencies.keys())}
    
    # Apply remmap in all data structures
    sentences = [delimiter.join([token
                                 for token in sentence.split(delimiter)
                                 if token in tokens])
                 for sentence in sentences]
    word_count = sum(frequencies.values())

    if verbose:
        print("Sentences:\t%d" % (len(sentences),))
        print("Tokens:\t\t%d" % (len(tokens),))
        print("Word Count:\t%d" % (word_count,))
        print("Corpora to Sentences completed")
        
    return sentences, tokens

"""
    Convert all articles to sentences
"""
    
def preprocess_command(verbose):
    parser = argparse.ArgumentParser(description='Process corpora')
    parser.add_argument('--corpora', dest='corpora_path', type=str,
                        help='path to corpora')
    parser.add_argument('--limit', dest='limit', type=int,
                        help='limit number of corpus')
    parser.add_argument('--stopwords', dest='stopwords_path', type=str,
                        help='path stopwords to be ignored while learning phrases')
    parser.add_argument('--sentences', dest='sentences_path', type=str,
                        help='path to store all sentences of processed articles')
    parser.add_argument('--tokens', dest='tokens_path', type=str,
                        help='path to store all tokens of processed articles')
    args, unknown = parser.parse_known_args()
    
    if verbose:
        print('Start processing corpora')

    # Load data
    corpora = []
    with open(args.corpora_path, mode="r", encoding="utf-8") as fp:
        for _ in range(args.limit):
            # Get line
            line = next(fp)

            # EOF
            if not line:
                break

            # Parse json and append
            corpus = json.loads(line)['text']
            corpora.append(corpus)

    # Run
    stopwords = [] if not args.stopwords_path else [line.strip() for line in open(args.stopwords_path)]
    sentences, tokens = corpora_to_sentences(corpora, stopwords, verbose=verbose)

    # Delete articles - memory restraint
    del corpora

    # Save
    with open(args.sentences_path, 'wb') as fp:
        pickle.dump(sentences, fp, protocol=pickle.HIGHEST_PROTOCOL) 
    with open(args.tokens_path, 'wb') as fp:
        pickle.dump(tokens, fp, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        print('Processing completed\n')

"""
    Filter Tokens
"""

def filter_command(verbose):
    parser = argparse.ArgumentParser(description='Filter sentences by frequency')
    parser.add_argument('--raw_sentences', dest='raw_sentences_path', type=str,
                        help='path to store all sentences of processed articles')
    parser.add_argument('--limit', dest='limit', type=int,
                        help='limit number of tokens')
    parser.add_argument('--sentences', dest='sentences_path', type=str,
                        help='path to store all sentences of processed articles')
    parser.add_argument('--tokens', dest='tokens_path', type=str,
                        help='path to store all tokens of processed articles')
    args, unknown = parser.parse_known_args()
    
    if verbose:
        print('Filtering vocabulary from sentences')
    
    # Load
    with open(args.raw_sentences_path, 'rb') as fp:
        raw_sentences = pickle.load(fp)

    # Run
    sentences, tokens = filter_vocabulary(raw_sentences, args.limit, verbose=verbose)

    # Save
    with open(args.sentences_path, 'wb') as fp:
        pickle.dump(sentences, fp, protocol=pickle.HIGHEST_PROTOCOL)
    with open(args.tokens_path, 'wb') as fp:
        pickle.dump(tokens, fp, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        print('Filtering completed\n')


"""
    Run experiments
"""

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Command selector')    
    parser.add_argument(dest='command', type=str,
                        help='Command to be used: preprocess; filter; split')    
    parser.add_argument('--verbose', dest='verbose', type=bool,
                        help='Print Progress')    
    args, unknown = parser.parse_known_args()
    
    if args.command == "preprocess":
        preprocess_command(args.verbose)
    elif args.command == "filter":
        filter_command(args.verbose)
    else:
        raise ValueError("Command %s unknown" % args.command)

