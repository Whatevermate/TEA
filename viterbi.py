
from nltk import word_tokenize,sent_tokenize
from math import floor,log,exp,ceil
from nltk.util import ngrams
from nltk.metrics import distance
from collections import Counter
import random
import string
import time



# Reads file
with open("europarl.txt", "r", encoding="utf8") as file:
    text = file.read()
    text = text.lower()

counter_replace, counter_add, counter_remove, counter_word, count_test_tokens,count_changed_chars = (0,0,0,0,0,0)

# Creates train set for bigrams 
train_set = sent_tokenize(text[0:floor(len(text)/15)])

# Creates development and test sets for bigrams 
dev_set = sent_tokenize(text[89*floor(len(text)/100):90*floor(len(text)/100)])
test_set = sent_tokenize(text[90*floor(len(text)/100):len(text)])


# Adds s1, e and replaces punctuation marks
datasets = (train_set, dev_set, test_set)
punctuation = ["?",'.','!',',', '"',"'", ";", ":", "(" , ")", "]", "[", "}", "{", "-"]

for dataset in datasets:  
    for i in range(len(dataset)):
        #Checks if s1 has already been inserted from previous runs
        if dataset[i][:4] != "*s1*":
            dataset[i] = "*s1* " + dataset[i]+ " *e* " 
            for mark in punctuation:
                dataset[i] = dataset[i].replace(mark, "")


# Creates tokens for bigrams from training set
train_set_tokens = [word_tokenize(i) for i in train_set]

train_set_tokens = [item for sublist in train_set_tokens for item in sublist]

# Creates a dictionary that records the frequency of each word
wordcount = Counter(train_set_tokens)
vocabulary = set()

# Creates a set containing all frequent vocabulary words (encountered more than 10 times) and
# replaces all infrequent words with the *UKN* token in the training set

for i in range(len(train_set_tokens)):
    if wordcount[train_set_tokens[i]] < 3:
        train_set_tokens[i] = "*UKN*"
    else:
        vocabulary.add(train_set_tokens[i])

vocabulary -= {'*s1*',"*e*" }
#remove all letters from vocabulary, except 'a'
for letter in string.ascii_lowercase[1:]:
    vocabulary -= {letter}
# remove all numbers from vocabulary 
new_vocabulary=set()   
for entry in vocabulary:
    if not entry.isdigit():
        new_vocabulary.add(entry)
vocabulary=new_vocabulary
del new_vocabulary

# Distinct words in vocabulary
v = len(vocabulary)


#datasets2 = (dev_set, test_set)

# Replaces infrequent words with *UKN* in dev and test sets
#for dataset in datasets2:
#    for i in range(len(dataset)):
#        for word in word_tokenize(dataset[i]):
#            if wordcount[word] < 10:
#                dataset[i] = dataset[i].replace(word, "*UKN*")

# Creates training set bigrams and trigrams
train_set_bigrams = [ gram for gram in ngrams(train_set_tokens, 2) ]


# Defines the Laplace function for bigrams
def laplace_bi(bigram, pool, text):
    return (pool.count(bigram) + 1) / (text.count(bigram[0]) + v)

# Implements the Viterbi dynamic algorithm
# sentence: the sentence as a string, depth: the number of possible candidates in each hidden state
def Viterbi(sentence, depth):
    sentence=sentence.lower()
    sentence_tokens=word_tokenize(sentence)
    candidates=candidate_dic(sentence_tokens,depth)
    viterbi_dict={}
    # Creates a backpointer in order to find the sequence once we've reached the end
    backpointer={}
    # The first viterbi value is known and equal to 1, its log is zero
    viterbi_dict[(0,'*s1*')]=0
    for state_index in range(1,len(sentence_tokens)):
        for word in candidates[sentence_tokens[state_index]]:
            # word is a tuple, the string is needed
            word=word[0]
            if state_index == len(sentence_tokens)-1:
                word = '*e*'
            # computes probability based on the levenshtein distance between current word and candidates
            leven_prob= 1/(1 + distance.edit_distance(word, sentence_tokens[state_index]))
            scores=[]
            #iterates over the candidates of the previous state to compute the bigram and viterbi scores
            for prev_word in candidates[sentence_tokens[state_index-1]]:
                prev_word = prev_word[0]
                if state_index == 1:
                    prev_word = '*s1*'
                bigram=(prev_word, word)
                laplace_prob=laplace_bi(bigram, train_set_bigrams, train_set_tokens)
                # retrieves the viterbi score of the previous state
                viterbi_prob = viterbi_dict[(state_index-1, prev_word)]
                score=log(laplace_prob) + viterbi_prob
                # appends a tuple of the previous word and the current score in a list
                scores.append((prev_word,score))
            # finds maximum score from the scores list
            maximum=sorted(scores, key=lambda position: -position[1])[0]
            # creates instance in the backpointer dictionary
            backpointer[(state_index, word)]=maximum[0]
            # appends current word and viterbi score in the viterbi dictionary
            viterbi_dict[(state_index, word)]=maximum[1]+log(leven_prob)       
    reverse_order=[]
    word_to_append = word
    # Reads word in the reverse order based on backpointer dictionary
    for i in range(len(sentence_tokens)-1,0,-1):
        reverse_order.append(word_to_append)
        word_to_append=backpointer[(i,word_to_append)]
    reverse_order.reverse()
    final_sequence=['*s1*']+ reverse_order
    # Computes sentence probability
    sequence_probability = exp(maximum[1])
    # Returns the actual sentence and its probability 
    return final_sequence, sequence_probability

 
# Implements the Levenshtein distance candidates
def candidate_dic(sentence_tokens,depth):
    levenshtein = {}
    candidates = {}
    # Iterates over word in the sentence skipping *s1*
    for word in sentence_tokens[1:len(sentence_tokens)]:
        if word not in levenshtein:
            levenshtein[word]=set()
        if word not in candidates:
            candidates[word]=set()
        for entry in vocabulary :
            levenshtein[word].add((entry, distance.edit_distance(word,entry)))
        #obtain the words with the smallest edit distance     
        candidates[word] = sorted(levenshtein[word], key=lambda word: word[1])[0:depth]
    if sentence_tokens[0]=='*s1*':
        candidates[sentence_tokens[0]] = sentence_tokens[0]
    if sentence_tokens[-1]=='*e*':
        candidates[sentence_tokens[-1]] = sentence_tokens[-1]
    return candidates

          
# Implements the baseline method, which is replacement based on minimum levenshtein distance
def baseline_method(sentence, depth=1):
    sentence=sentence.lower()
    sentence_tokens=word_tokenize(sentence)
    candidates=candidate_dic(sentence_tokens,depth)
    sequence=['*s1*']
    probability=1
    for state_index in range(1,len(sentence_tokens)):
        scores=[]
        for word in candidates[sentence_tokens[state_index]]:
            # word is a tuple, the string is needed
            word=word[0]
            if state_index == len(sentence_tokens)-1:
                word = '*e*'
            # computes probability based on the levenshtein distance between current word and candidates
            leven_prob= 1/(1 + distance.edit_distance(word, sentence_tokens[state_index]))
            prev_word = sequence[state_index-1]
            bigram=(prev_word, word)
            laplace_prob=laplace_bi(bigram, train_set_bigrams, train_set_tokens)
            score=log(laplace_prob) + log(leven_prob)+log(probability)
            # appends a tuple of the previous word and the current score in a list
            scores.append((word,score))
        # finds maximum score from the scores list
        maximum=sorted(scores, key=lambda position: -position[1])[0]
        sequence.append(maximum[0])
        probability=exp(maximum[1])
    # Returns the actual sentence and its probability 
    return sequence, probability


# Implements the evaluation metric, which is percentage of common words between two correct sentence and correctED sentence
def evaluation_metrics(correct_sent, proposed_sent, wrong_sents):
    correct = correct_sent[:]
    proposed = proposed_sent[:]
    wrong=wrong_sents[:]
    same = 0
    total = 0
    could_not_find=0
    wrong_correction=0
    for index in range(len(correct_sent)):
        proposed[index] = word_tokenize(proposed_sent[index])
        correct[index] = word_tokenize(correct_sent[index])
        wrong[index]=word_tokenize(wrong_sents[index])
        total += len(correct[index])
        len(proposed[index])
        for word_index in range(len(proposed[index])):
            # If the word is unknown, consider the suggestion to be correct
            if proposed[index][word_index] == correct[index][word_index]:
                same += 1
            elif correct[index][word_index] not in vocabulary:
                could_not_find+=1
            if (proposed[index][word_index] != correct[index][word_index]) and (wrong[index][word_index] == correct[index][word_index]):
                wrong_correction+=1
    eval_score = same/total
    not_in_vocab=could_not_find/total
    wrongly_corrected=wrong_correction/total
    return eval_score,not_in_vocab,wrongly_corrected

# Implements an introduction of random mistakes in a given sentence
# Decides between 0 and len(sentence) number of mistakes
# Considers two kind of mistakes: change letter and change word which are implemented in other functions
# 
def make_mistakes(sentence, seed = 1):
    random.seed(seed)
    init_tokens = word_tokenize(sentence)
    global count_test_tokens
    global counter_word
    count_test_tokens += len(init_tokens)
    tokens = init_tokens[:]
    num_mistakes = random.randint(0, len(tokens)-1)
    indices = list(range(1,len(tokens)))
    picked = random.sample(indices, num_mistakes)
    for item in picked:
        word_to_convert = tokens[item]
        if word_to_convert != "*UKN*":
            coin_toss = random.randint(0, 1)
            if coin_toss != 0:
                converted_word = change_letter(word_to_convert)
                tokens[item] = converted_word
            else:
                converted_word = change_word(word_to_convert)
                counter_word+=1
                tokens[item] = converted_word
        else:
            continue
    converted_sent = ' '.join(tokens)
    return converted_sent

##############################################################################                
##############################################################################                





##############################################################################                
##############################################################################                

def change_letter(word):
    methods = ['replace', 'remove', 'add']
    method = random.sample(methods, 1)[0]
    times = random.sample(list(range(1+ceil(len(word)/3))).remove(0), 1)[0]	
    global counter_replace, counter_remove, counter_add,count_changed_chars
    count_changed_chars+=times
    if method == 'replace':
        converted_word = replace_letter(word,times)
        counter_replace+=1
    elif method == 'remove':
        converted_word = remove_letter(word,times)
        counter_remove+=1
    elif method == 'add':
        converted_word = add_letter(word,times)
        counter_add+=1
    else:
        return 'Method selection error'
    return converted_word
##############################################################################                
##############################################################################                





##############################################################################                
##############################################################################                

def replace_letter(word,times):
#    random.seed(seed)
    indices = list(range(len(word)))
    picked = random.sample(indices, times)
    converted_word=word
    for index in range(len(picked)):  
        char1 = random.choice(string.ascii_lowercase) 
        while char1 == word[picked[index]]:
            char1=random.choice(string.ascii_lowercase)
        converted_word = converted_word.replace(word[picked[index]], char1, 1)
    return(converted_word)

##############################################################################                
##############################################################################                






##############################################################################                
##############################################################################                

def remove_letter(word,times):
#    random.seed(seed)
    tokens = list(word)
    indices = list(range(len(tokens)))
    picked = random.sample(indices, times)
    converted_word=word
    for index in range(len(picked)):  
        converted_word = converted_word.replace(word[picked[index]], '', 1)
    return(converted_word)

##############################################################################                
##############################################################################                





##############################################################################                
##############################################################################                

def add_letter(word,times):
    tokens = list(word)
    indices = list(range(len(tokens)))
    picked = random.sample(indices, times)
    picked=sorted(picked)
    increase=0
    for index in range(len(picked)):  
        char1 = random.choice(string.ascii_lowercase) 
        tokens.insert(picked[index]+increase, char1)
        increase+=1
    converted_word = ''.join(tokens)
    return(converted_word)

##############################################################################                
##############################################################################                



     
def change_word(word):
    pool=candidate_dic(['nonsense',word],20)[word] 
    new_word = random.sample(list(pool), 1)[0]
    return new_word[0]
##############################################################################                
##############################################################################                
   

##############################################################################                
##############################################################################                



start = time.time()
random.seed(45)
test_sentences = random.sample(test_set, 20)  
viterbi_sents = []
baseline_sents=[]
new_sents = []
i = 0
viterbi_time=0
baseline_time=0
for sent in test_sentences:
    i+=1
    new_sent = make_mistakes(sent, seed = i)
    new_sents.append(new_sent)
    start_vit = time.time()
    viterbi_sents.append(' '.join(Viterbi(new_sent,10)[0]))
    end_vit = time.time()
    viterbi_time+=end_vit - start_vit
    start_base = time.time()
    baseline_sents.append(' '.join(baseline_method(new_sent,10)[0]))
    end_base = time.time()
    baseline_time+=end_base-start_base


    
end = time.time()
elapsed = end - start
print(elapsed)

initial_score=evaluation_metrics(test_sentences,new_sents, new_sents)
viterbi_score=evaluation_metrics(test_sentences,viterbi_sents, new_sents)
baseline_score=evaluation_metrics(test_sentences,baseline_sents, new_sents)

total_chars=0
for sentence in test_sentences:
    new_sent=word_tokenize(sentence)
    for word in new_sent:
        for char in word:
            total_chars+=1

print("Percentage of replaced words: {}\n".format(counter_word/count_test_tokens))

print("Percentage of words with removed tokens: {}\n".format(counter_remove/count_test_tokens))

print("Percentage of words with inserted tokens: {}\n".format(counter_add/count_test_tokens))

print("Percentage of words with replaced tokens: {}\n".format(counter_replace/count_test_tokens))

print("Percentage of different characters: {}\n".format(count_changed_chars/total_chars))

print("Percentage of different words between the test sentences and the sentences with mistakes: {}\n".format(initial_score[0]))

print("Percentage of different words between the test sentences and the sentences produced by the viterbi method: {}\n".format(viterbi_score[0]))

print("Percentage of different words between the test sentences and the sentences produced by the baseline method: {}\n".format(baseline_score[0]))

print("Percentage of wrongly predicted words, the correct version of which was not in the vocabulary and therefore viterbi could not predict: {}\n".format(viterbi_score[1]))

print("Percentage of wrongly predicted words, the correct version of which was not in the vocabulary and therefore baseline could not predict: {}\n".format(baseline_score[1]))

print("Percentage of wrongly predicted words by the viterbi method that had not sustained error introduction: {}\n".format(viterbi_score[2]))

print("Percentage of wrongly predicted words by the baseline method that had not sustained error introduction: {}\n".format(baseline_score[2]))



i=0
print("Original sentence: ")
print(test_sentences[i] + '\n')
print("Sentence with random mistakes: ")
print(new_sents[i]+'\n')
print("Corrected with Viterbi: ")
print(viterbi_sents[i] + '\n')
print("Corrected with baseline: ")
print(baseline_sents[i] + '\n')


