import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

paragraph = """What is Round-Robin Scheduling?
Characteristics of Round-Robin Scheduling
Example of Round-robin Scheduling
Advantage of Round-robin Scheduling
Disadvantages of Round-robin Scheduling
Worst Case Latency
Characteristics of Round-Robin Scheduling
Here are the important characteristics of Round-Robin Scheduling:
Round robin is a pre-emptive algorithm
The CPU is shifted to the next process after fixed interval time, which is called time quantum/time slice.
The process that is preempted is added to the end of the queue.
Round robin is a hybrid model which is clock-driven
Time slice should be minimum, which is assigned for a specific task that needs to be processed. However, it may differ OS to OS.
It is a real time algorithm which responds to the event within a specific time limit.
Round robin is one of the oldest, fairest, and easiest algorithm.
Widely used scheduling method in traditional OS.
Example of Round-robin Scheduling
Consider this following three processes """

sentences =  nltk.sent_tokenize(paragraph)
stemmer = PorterStemmer()

#Stemming

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(word) for word in words if word not in set(stopwords.words('english'))]
    sentences[i] = ''.join(words)