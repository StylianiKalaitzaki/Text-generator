import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
import string
from collections import Counter
import random
nltk.download('gutenberg')# Download the corpus if not already available
nltk.download('punkt')

def load_books():
    """
    Load the first 10 books from the Project Gutenberg corpus using NLTK.

    Returns:
    List of strings: Texts of the first 10 books.
    """
    # Get the file IDs of the last 10 books
    book_ids = gutenberg.fileids()[-10:]

    # Combine the texts from the selected books into a single document
    combined_text = ""
    for fileid in book_ids:
      combined_text += gutenberg.raw(fileid)

    return combined_text

def create_word_dictionary(text):
  """
    Creates a dictionary of word frequencies from the given text.

    Args:
    text (str): The input text from which to generate the word dictionary.

    Returns:
    dict: A dictionary where keys are unique words and values are their frequencies in the text.
    """

  # Tokenize the text into words
  tokens = word_tokenize(text)

  # Remove punctuation from tokens
  tokens = [token.lower() for token in tokens if token.isalpha()]

  word_freq = Counter(tokens)

  return word_freq

def generate_ngrams(sentences, n):
    """
    Generate n-grams from sentences.

    Args:
    - sentences: List of sentences
    - n: Length of n-grams

    Returns:
    - ngrams: List of n-grams
    """
    ngrams = []
    for sentence in sentences:
        words = ["<s>"]*(n-1) + word_tokenize(sentence.lower()) + ["</s>"]*(n-1)
        ngrams.extend(list(nltk.ngrams(words, n)))
    return ngrams

def generate_sent(ngram_freq, n):
  # Choose a starting word
  starting_word = "<s>"

  # Initialize the sentence
  sentence = [starting_word]*(n-1)


  # Generate the sentence
  while True:
      # Get the last n-1 words in the sentence
      last_n_words = tuple(sentence[-(n - 1):])

      # Stop if the last word is a sentence-ending token
      if last_n_words[-1] == "</s>":
          break

      # Get all n-grams starting with the last n-1 words
      possible_ngrams = [ngram for ngram in ngram_freq if ngram[:-1] == last_n_words]

      # If there are no ngrams starting with the last n-1 words, end the sentence
      if not possible_ngrams:
          break

      # Calculate the total frequency of all possible ngrams
      total_frequency = sum(ngram_freq[ngram] for ngram in possible_ngrams)

      # Choose a random number between 0 and the total frequency
      rand_num = random.uniform(0, total_frequency)

      # Choose the ngram based on the random number and its cumulative frequency
      cumulative_frequency = 0
      for ngram in possible_ngrams:
          cumulative_frequency += ngram_freq[ngram]
          if rand_num <= cumulative_frequency:
              chosen_ngram = ngram
              break


      # Add the third word of the chosen trigram to the sentence
      sentence.append(chosen_ngram[n-1])



  # Join the words in the sentence to form the final output
  #generated_sentence = ' '.join(sentence[n-1:-(n-1)])  # Exclude the start and end tokens
  generated_sentence = ' '.join(sentence[n-2:])
  return generated_sentence
  
text = load_books()

word_dict = create_word_dictionary(text)

print("Vocabulary length: ", len(word_dict, "\n"))
# Print the 10 most frequent words
print("10 Most frequent words:")
for word, frequency in word_dict.most_common(10):
    print(word, ":", frequency)

# Sentence tokenization breaks the text into a list of sentences.
sentences = nltk.sent_tokenize(text)

# Generate n-grams
unigrams = generate_ngrams(sentences, 1)
bigrams = generate_ngrams(sentences, 2)
trigrams = generate_ngrams(sentences, 3)
tetragrams = generate_ngrams(sentences, 4)

# Calculate frequencies
unigram_freq = Counter(unigrams)
bigram_freq = Counter(bigrams)
trigram_freq = Counter(trigrams)
tetragram_freq = Counter(tetragrams)

# Generate 5 senteces for each ngram
bi_sents = ""
tri_sents = ""
tetr_sents = ""
for i in range(5):
  bi_sents += generate_sent(bigram_freq,2)+'\n'
  tri_sents += generate_sent(trigram_freq,3)+'\n'
  tetr_sents += generate_sent(tetragram_freq,4)+'\n'

print("\n5 senteces using Bigrams:")
print(bi_sents)
print("\n")
print("5 senteces using Trigrams:")
print(tri_sents)
print("\n")
print("5 senteces using Tetragrams:")
print(tetr_sents)
