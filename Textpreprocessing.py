#preprocessing captions
import re
from nltk.corpus import stopwords
import Datacollection

def cleantext(rev):
  rev = re.sub(r'[^a-zA-Z]',' ',rev)
  rev = rev.lower()
  rev = rev.split()
  rev = [word for word in rev if len(word)>1]
  rev = [word for word in rev if word.isalpha()]
  rev = ' '.join(rev)
  rev = 'startseq ' + rev + ' endseq'
  return rev
for i in range(len(result1)):
  result2[i][1] = cleantext(result2[i][1])
from keras.preprocessing.text import Tokenizer
max_features = 10000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(result2[:,1])
list_tokenized_train = tokenizer.texts_to_sequences(result2[:,1])
num_words = len(tokenizer.word_index) + 1
print(num_words)
