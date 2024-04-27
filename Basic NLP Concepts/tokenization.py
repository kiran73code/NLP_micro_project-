import nltk
nltk.download('punkt')
sentence = """Spyder is an Integrated Development Environment (IDE) for scientific computing, written in and for the Python programming language. It comes with an Editor to write code, a Console to evaluate it and view the results at any time, a Variable Explorer to examine the variables defined during evaluation, and several other facilities to help you effectively develop the programs you need as a scientist.

This tutorial was originally authored by Hans Fangohr from the University of Southampton (UK), and subsequently updated for Spyder 3.3.x by the development team (see Historical note for more details)."""
 

#divided into sentences sentence tokenization
sentences = nltk.sent_tokenize(sentence)
print(sentences)


# word tokenization
words = nltk.word_tokenize(sentence)
print(words)