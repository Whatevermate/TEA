from nltk.tokenize import RegexpTokenizer
from nltk import word_tokenize
from nltk import sent_tokenize
from math import floor,log,exp
from nltk.util import ngrams
from nltk.metrics import distance
from numpy import product as  prod
from collections import Counter
from math import log, exp
import random

# Reads file
with open("europarl.txt", "r", encoding="utf8") as file:
    text = file.read()
    text = text.lower()


# Creates train set for bigrams and trigrams
train_set2 = sent_tokenize(text[0:floor(len(text)/50)])
train_set3 = train_set2[:]

# Creates development and test sets for bigrams and trigrams
dev_set2 = sent_tokenize(text[89*floor(len(text)/100):90*floor(len(text)/100)])
dev_set3 = dev_set2[:]

test_set2 = sent_tokenize(text[99*floor(len(text)/100):len(text)])
test_set3 = test_set2[:]


# Adds s1, s2, e and replaces punctuation marks

datasets2 = (train_set2, dev_set2, test_set2)
datasets3 = (train_set3, dev_set3, test_set3)
punctuation = ["?",'.','!',',', '"',"'", ";", ":", "(" , ")", "]", "[", "}", "{", "-"]

for dataset in datasets2:  
    for i in range(len(dataset)):
        #Checks if s1 has already been inserted from previous runs
        if dataset[i][:4] != "*s1*":
            dataset[i] = "*s1* " + dataset[i] + " *e*"
            for mark in punctuation:
                dataset[i] = dataset[i].replace(mark, "")

for dataset in datasets3:                
    for i in range(len(dataset)):
        #Checks if s1 has already been inserted from previous runs
        if dataset[i][:4] != "*s1*":
            dataset[i] = "*s1* *s2* " + dataset[i] + " *e*"
            for mark in punctuation:
                dataset[i] = dataset[i].replace(mark, "")


# Creates tokens for bigrams and trigrams from training set
train_set2_tokens = [word_tokenize(i) for i in train_set2]
train_set3_tokens = [word_tokenize(i) for i in train_set3]

train_set2_tokens = [item for sublist in train_set2_tokens for item in sublist]
train_set3_tokens = [item for sublist in train_set3_tokens for item in sublist]

# Creates a dictionary that records the frequency of each word
wordcount = Counter(train_set3_tokens)
vocabulary = set()

# Creates a set containing all frequent vocabulary words (encountered more than 10 times) and
# replaces all infrequent words with the *UKN* token in the training set

for i in range(len(train_set2_tokens)):
    if wordcount[train_set2_tokens[i]] < 10:
        train_set2_tokens[i] = "*UKN*"
    else:
        vocabulary.add(train_set2_tokens[i])
        
for i in range(len(train_set3_tokens)):
    if wordcount[train_set3_tokens[i]] < 10:
        train_set3_tokens[i] = "*UKN*"

vocabulary -= {'*s1*', '*end*', '*s2*'}

# Distinct words in vocabulary
v = len(vocabulary)


datasets = (dev_set2, dev_set3, test_set2, test_set3)

# Replaces infrequent words with *UKN* in dev and test sets
for dataset in datasets:
    for i in range(len(dataset)):
        for word in word_tokenize(dataset[i]):
            if wordcount[word] < 10:
                dataset[i] = dataset[i].replace(word, "*UKN*")



# Creates training set bigrams and trigrams

train_set_bigrams = [ gram for gram in ngrams(train_set2_tokens, 2) ]
train_set_trigrams = [ gram for gram in ngrams(train_set3_tokens, 3) ]


# Defines the Laplace function for bigrams and trigrams

def laplace_bi(bigram, pool, text):
    return (pool.count(bigram) + 1) / (text.count(bigram[0]) + v)
    
def laplace_tri(trigram, pool1, pool2):
    return (pool1.count(trigram) + 1) / (pool2.count(trigram[0:2]) + v)


random.seed(15)

# Creates correct (encountered) and incorrect (random tokens from vocabulary) sentences, 5 words long, to test the model

correct_sents2 = random.sample(dev_set2, 5)
incorrect_sents2 = [random.sample(vocabulary, len(word_tokenize(i))) for i in correct_sents2]
correct_sents3 = random.sample(dev_set3, 5)
incorrect_sents3 = [random.sample(vocabulary, len(word_tokenize(i))) for i in correct_sents3]

for i in range(len(correct_sents2)):
    incorrect_sents2[i] = ' '.join(incorrect_sents2[i])
    
for i in range(len(incorrect_sents3)):
    incorrect_sents3[i] = ' '.join(incorrect_sents2[i])


# Implements the model to calculate probabilities of each sentence

bi_probs_correct = []
bi_probs_incorrect = []

tri_probs_correct = []
tri_probs_incorrect = []

for i in range(len(correct_sents2)):
    correct_sents2_tokens = word_tokenize(correct_sents2[i])
    incorrect_sents2_tokens = ['*s1*'] + word_tokenize(incorrect_sents2[i]) + ['*e*']

    correct_bigrams = [ gram for gram in ngrams(correct_sents2_tokens, 2) ]
    incorrect_bigrams = [ gram for gram in ngrams(incorrect_sents2_tokens, 2) ]
    
    sum_correct = 0
    sum_incorrect = 0
    
    for j in range(len(correct_bigrams)):
        sum_correct += log(laplace_bi(correct_bigrams[j], train_set_bigrams, train_set2_tokens))
        sum_incorrect += log(laplace_bi(incorrect_bigrams[j], train_set_bigrams, train_set2_tokens))
        
    bi_probs_correct.append(exp(sum_correct))
    bi_probs_incorrect.append(exp(sum_incorrect))
    
for i in range(len(correct_sents2)):
    sum_correct = 0
    sum_incorrect = 0
    
    correct_sents3_tokens = word_tokenize(correct_sents3[i])
    incorrect_sents3_tokens = ['*s1*', '*s2*'] + word_tokenize(incorrect_sents3[i]) + ['*e*']
    
    correct_trigrams = [ gram for gram in ngrams(correct_sents3_tokens, 3) ]
    incorrect_trigrams = [ gram for gram in ngrams(incorrect_sents3_tokens, 3) ] + ['*e*']
    
    for k in range(len(correct_trigrams)):
        sum_correct += log(laplace_tri(correct_trigrams[k], train_set_trigrams, train_set3_tokens))
        sum_incorrect += log(laplace_tri(incorrect_trigrams[k], train_set_trigrams, train_set3_tokens))
        
    tri_probs_correct.append(exp(sum_correct))
    tri_probs_incorrect.append(exp(sum_incorrect))
    
print("Probabilities of correct sentences as predicted by the bigram model:")    
print(bi_probs_correct)
print("Probabilities of correct sentences as predicted by the trigram model:")  
print(tri_probs_correct)
print("Probabilities of incorrect sentences as predicted by the bigram model:")   
print(bi_probs_incorrect)
print("Probabilities of incorrect sentences as predicted by the trigram model:") 
print(tri_probs_incorrect)


# Implements a prediction for a given word sequence. Returns the predicted word and its probability
# Bigram model

trial_sent = 'you want'
trial_token = word_tokenize(trial_sent)
 
max_bi = 0
for word in vocabulary:
    trial_bi = (trial_token[-1], word)
    prob = laplace_bi(trial_bi, train_set_bigrams, train_set2_tokens)
    if prob > max_bi:
        max_bi = prob
        best_choice_bi = word

print("'" + best_choice_bi + "' with a probability of " + str(max_bi))


# Trigram Model

max_tri = 0
for word in vocabulary:
    trial_tri = (trial_token[-2], trial_token[-1], word)
    prob = laplace_tri(trial_tri, train_set_trigrams, train_set3_tokens)
    if prob > max_tri:
        max_tri = prob
        best_choice_tri = word

print("'" + best_choice_tri + "' with a probability of " + str(max_tri))


# Calculates the cross entropy and perplexity of the bigram model for the whole development set

probabilities = []

for sentence in dev_set2:
    dev_set_tokens_bi = word_tokenize(sentence)
    dev_set_bigrams = [ gram for gram in ngrams(dev_set_tokens_bi, 2) ]
    for bigram in dev_set_bigrams:
        probabilities.append(laplace_bi(bigram, train_set_bigrams, train_set2_tokens))
        
sum1 = 0

for probability in probabilities:
    sum1 += log(probability, 2)

cross_entropy_bigram = (-1 / len(probabilities)) * sum1
perplexity_bigram = 2**cross_entropy_bigram


# Calculates the cross entropy and perplexity of the trigram model for the whole development set

probabilities_tri = []

for sentence in dev_set3:
    dev_set_tokens_tri = word_tokenize(sentence)
    dev_set_trigrams = [ gram for gram in ngrams(dev_set_tokens_tri, 2) ]
    for trigram in dev_set_trigrams:
        probabilities_tri.append(laplace_tri(trigram, train_set_trigrams, train_set3_tokens))
        
sum1_tri = 0

for probability in probabilities_tri:
    sum1_tri += log(probability, 2)

cross_entropy_trigram = (-1 / len(probabilities_tri)) * sum1_tri
perplexity_trigram = 2**cross_entropy_trigram


print("Cross entropy of the bigram model: {}".format(cross_entropy_bigram))
print("Perplexity of the bigram model: {}".format(perplexity_bigram))

print("Cross entropy of the trigram model: {}".format(cross_entropy_trigram))
print("Perplexity of the trigram model: {}".format(perplexity_trigram))

