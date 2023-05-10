import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Define the text paragraph
text = "Natural Language Processing (NLP) is a field of study focused on the interactions between human language and computers. It is used to program computers to process and analyze large amounts of natural language data. NLP is used in a variety of applications, including machine translation, sentiment analysis, and speech recognition."

# Convert the text to lowercase
text = text.lower()

# Tokenize the text into sentences
sentences = sent_tokenize(text)

# Remove stopwords from the sentences
stop_words = set(stopwords.words('english'))
filtered_sentences = []
for sentence in sentences:
    words = word_tokenize(sentence)
    filtered_sentence = [word for word in words if word.casefold() not in stop_words]
    filtered_sentences.append(' '.join(filtered_sentence))

# Stem the words in the sentences
stemmer = PorterStemmer()
stemmed_sentences = []
for sentence in filtered_sentences:
    words = word_tokenize(sentence)
    stemmed_sentence = [stemmer.stem(word) for word in words]
    stemmed_sentences.append(' '.join(stemmed_sentence))

# Lemmatize the words in the sentences
lemmatizer = WordNetLemmatizer()
lemmatized_sentences = []
for sentence in stemmed_sentences:
    words = word_tokenize(sentence)
    lemmatized_sentence = [lemmatizer.lemmatize(word) for word in words]
    lemmatized_sentences.append(' '.join(lemmatized_sentence))

# Print the preprocessed sentences
for sentence in lemmatized_sentences:
    print(sentence)
