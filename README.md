# nlp-trigram-model

## COMS W4705 - NLP 
Homework 1 - Building a Trigram Language Model

### Functions:
1. 'corpus_reader' 'get_lexicon' to read corpus and define lexicon.
2. 'get_ngrams' 'count_ngrams' to extract n-grams from a sentence and calculate the number of each token.
3. 'raw_trigram_probability', 'raw_bigram_probability', 'raw_unigram_probability' to calculate the unsmoothed probability, which can be used to generate random sentence using function 'generate_sentence'.
4. 'smoothed_trigram_probability', 'sentence_logprob', 'perplexity' are used to calculate the perplexity of the test corpus, which is considered to be the accuracy of classification.

### Examples:
1. Using brown dataset to build the model and calculate the perplexity.
2. Using the old classified scores (high, low) of ETS TOEFL to build the trigram model, use it to calculate the perplexity of the test classified scores, and then get the accuracy of the trigram model.
