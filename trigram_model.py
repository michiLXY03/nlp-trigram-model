import sys
from collections import defaultdict
import math
import random
import os
import os.path
import numpy as np
"""
COMS W4705 - Natural Language Processing - Spring 2020
Homework 1 - Programming Component: Trigram Language Models
Yassine Benajiba
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    """
    COMPLETE THIS FUNCTION (PART 1)
    Given a sequence, this function should return a list of n-grams, where each n-gram is a Python tuple.
    This should work for arbitrary values of 1 <= n < len(sequence).
    """
    if (not 1 <= n < len(sequence)):
        return []
    
    sequence = ['START'] * ((n-1) if n > 1 else 1) + sequence + ['END']
    res = zip(*[sequence[_:] for _ in range(n)])

    return res


class TrigramModel(object):
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):
        """
        COMPLETE THIS METHOD (PART 2)
        Given a corpus iterator, populate dictionaries of unigram, bigram,
        and trigram counts. 
        """
   
        self.unigramcounts = {} # might want to use defaultdict or Counter instead
        self.bigramcounts = {} 
        self.trigramcounts = {} 

        ##Your code here
        self.word_num = 0

        for seq in corpus:
            unigrams = get_ngrams(seq, 1)
            bigrams = get_ngrams(seq, 2)
            trigrams = get_ngrams(seq, 3)
            k_ss = ('START', 'START')

            for key in list(unigrams):
                self.unigramcounts.update({key: self.unigramcounts[key]+1 if self.unigramcounts.get(key, 0) else 1})

            for key in list(bigrams):
                self.bigramcounts.update({key: self.bigramcounts[key]+1 if self.bigramcounts.get(key, 0) else 1})

            for key in list(trigrams):
                self.trigramcounts.update({key: self.trigramcounts[key]+1 if self.trigramcounts.get(key, 0) else 1})
                if key[:2] == k_ss:
                    self.bigramcounts.update({k_ss: self.bigramcounts[k_ss]+1 if self.bigramcounts.get(k_ss, 0) else 1})

        self.word_num = sum(self.unigramcounts.values()) - self.unigramcounts[('START',)] - self.unigramcounts[('END',)]
                
        return
    
    def raw_trigram_probability(self,trigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) trigram probability
        """
        if trigram not in self.trigramcounts:
            return 0.0
        
        if (self.bigramcounts[trigram[:2]] != 0):
            return self.trigramcounts[trigram]/self.bigramcounts[trigram[:2]]
        
        return 0.0

    
    def raw_bigram_probability(self, bigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) bigram probability
        """
        if bigram not in self.bigramcounts:
            return 0.0
        
        if (self.unigramcounts[bigram[:1]] != 0):
            return self.bigramcounts[bigram]/self.unigramcounts[bigram[:1]]
        
        return 0.0
    
    
    def raw_unigram_probability(self, unigram):
        """
        COMPLETE THIS METHOD (PART 3)
        Returns the raw (unsmoothed) unigram probability.
        """
        if unigram not in self.unigramcounts:
            return 0.0
        
        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.
        
        # self.word_num had been calculated in function count_ngrams
        return self.unigramcounts[unigram]/self.word_num

    def generate_sentence(self,t=20): 
        """
        COMPLETE THIS METHOD (OPTIONAL)
        Generate a random sentence from the trigram model. t specifies the
        max length, but the sentence may be shorter if STOP is reached.
        """
        #Should import numpy as np
        result = []
        gram = (None,'START', 'START')
        
        while (gram[2] != 'STOP' and len(result) < t):
            word_1 = gram[2]
            word_2 = gram[1]

            next_grams = [trigram for trigram in self.trigramcounts if trigram[1] == word_1 and trigram[0] == word_2]
            if len(next_grams) == 0:
                break
            words = [trigram[2] for trigram in next_grams]
                
            prob = np.array([self.raw_trigram_probability(trigram) for trigram in next_grams])
            prob = prob/np.sum(prob)
            
            choice_list = np.random.choice(words, 1, p=prob)
            
            if choice_list.shape[0] > 0:
                new_word = choice_list[0]
                result.append(new_word)
                gram = (word_2, word_1, new_word)
            else:
                break
        
        if result[-1] == 'END':
            return result[:-1]
        
        return result   

    
    def smoothed_trigram_probability(self, trigram):
        """
        COMPLETE THIS METHOD (PART 4)
        Returns the smoothed trigram probability (using linear interpolation). 
        """
        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        
        res = 0.0
        res = lambda1 * self.raw_trigram_probability(trigram)\
             +lambda2 * self.raw_bigram_probability(trigram[1:])\
             +lambda3 * self.raw_unigram_probability(trigram[2:])
            
        return res
    

    def sentence_logprob(self, sentence):
        """
        COMPLETE THIS METHOD (PART 5)
        Returns the log probability of an entire sequence.
        """
        trigrams = get_ngrams(sentence, 3)
        s_prob = [self.smoothed_trigram_probability(gram) for gram in trigrams]
        g = lambda x: math.log2(x) if x > 0 else 0
        res = sum(list(map(g,s_prob)))
#         print(res)
        
        return res
    
    
    def perplexity(self, corpus):
        """
        COMPLETE THIS METHOD (PART 6) 
        Returns the log probability of an entire sequence.
        """
        #unigram_test = get_ngrams(corpus, 1)
        l = 0
        M = 0
        for sentence in corpus:
            l += self.sentence_logprob(sentence)
            M += len(sentence)

        return 2**(-l/M)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            # .. 
            pp_false = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            total += 1
            correct += 1 if pp < pp_false else 0
    
        for f in os.listdir(testdir2):
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            # .. 
            pp_false = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            total += 1
            correct += 1 if pp < pp_false else 0
        
        return correct/total

if __name__ == "__main__":

    model = TrigramModel(sys.argv[1]) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 

    
    # Testing perplexity: 
    # dev_corpus = corpus_reader(sys.argv[2], model.lexicon)
    # pp = model.perplexity(dev_corpus)
    # print(pp)


    # Essay scoring experiment: 
    # acc = essay_scoring_experiment('train_high.txt', 'train_low.txt", "test_high", "test_low")
    # print(acc)

    ## TEST 1 (Brown dataset)
    corpusfile = "./data/brown_train.txt"
    corpusfile_test = "./data/brown_test.txt"
    model_brown = TrigramModel(corpusfile)
    # perplexity
    pp_brown = model_brown.perplexity(corpus_reader(corpusfile_test, model_brown.lexicon))
    print(pp_brown)
    # generat new sentence
    model_brown.generate_sentence(20)

    ## TEST 2 (ETS_TOEFL dataset)
    training_file1 = "./data/ets_toefl_data/train_high.txt"
    training_file2 = "./data/ets_toefl_data/train_low.txt"
    testdir1 = "./data/ets_toefl_data/test_high"
    testdir2 = "./data/ets_toefl_data/test_low"
    acc = essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2)
    print(acc)


