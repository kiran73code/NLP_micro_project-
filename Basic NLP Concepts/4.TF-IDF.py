import nltk
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

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
Consider this following three processes It doesn't face the issues of starvation or convoy effect.
All the jobs get a fair allocation of CPU.
It deals with all process without any priority
If you know the total number of processes on the run queue, then you can also assume the worst-case response time for the same process.
This scheduling method does not depend upon burst time. That's why it is easily implementable on the system.
Once a process is executed for a specific set of the period, the process is preempted, and another process executes for that given time period.
Allows OS to use the Context switching method to save states of preempted processes.
It gives the best performance in terms of average response time.

Disadvantages of Round-robin Scheduling
Here, are drawbacks/cons of using Round-robin scheduling:

If slicing time of OS is low, the processor output will be reduced.
This method spends more time on context switching
Its performance heavily depends on time quantum.
Priorities cannot be set for the processes.
Round-robin scheduling doesn't give special priority to more important tasks.
Decreases comprehension
Lower time quantum results in higher the context switching overhead in the system.
Finding a correct time quantum is a quite difficult task in this system.
Worst Case Latency
This term is used for the maximum time taken for execution of all the tasks.

dt = Denote detection time when a task is brought into the list
st = Denote switching time from one task to another
et = Denote task execution time
Formula:

Tworst = {(dti+ sti + eti ), + (dti+ sti + eti )2 +...+ (dti+ sti + eti )N., + (dti+ sti + eti  + eti) N} + tISR	
t,SR = sum of all execution times
Summary:
The name of this algorithm comes from the round-robin principle, where each person gets an equal share of something in turns.
Round robin is one of the oldest, fairest, and easiest algorithms and widely used scheduling methods in traditional OS.
Round robin is a pre-emptive algorithm
The biggest advantage of the round-robin scheduling method is that If you know the total number of processes on the run queue, then you can also assume the worst-case response time for the same process.
This method spends more time on context switching
Worst-case latency is a term used for the maximum time taken for the execution of all the tasks."""


ps =PorterStemmer()
lemmatizer = WordNetLemmatizer()
sentences =  nltk.sent_tokenize(paragraph)



#Cleaning data


corpus =[]
for i in range(len(sentences)):
    review = re.sub('[^a-zA-Z]', ' ',sentences[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review=' '.join(review)
    corpus.append(review)
    
#Creating the bag of words
from sklearn.feature_extraction.text import TfidfVectorizer
cv =TfidfVectorizer()
x = cv.fit_transform(corpus).toarray()
