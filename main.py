import nltk

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import wikipedia

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')


lemmatizer = WordNetLemmatizer()
def lemma_me(sent):
    sent_tockens = nltk.word_tokenize(sent.lower())
    pos_tags = nltk.pos_tag(sent_tockens)
    sentlemma=[]
    for tocken,tag in zip(sent_tockens,pos_tags):
       if tag[1][0].lower() in  ['v','n','a','r']:
           lemma = lemmatizer.lemmatize(tocken, tag[1][0].lower())
           sentlemma.append(lemma)
    return sentlemma

topic = input('What topic do you want to ask about?\n')
text = wikipedia.page(topic).content

def process(text,question):
  sentence_tokens = nltk.sent_tokenize(text)
  sentence_tokens.append(question)

  tv = TfidfVectorizer(tokenizer=lemma_me)
  tf=tv.fit_transform(sentence_tokens)
  tf.toarray()
  values = cosine_similarity(tf[-1], tf)
  index = values.argsort()[0][-2]
  values_flat = values.flatten()
  values_flat.sort()
  coeff = values_flat[-2]
  if coeff>0.3:
    return sentence_tokens[index]

while True:
  question = input('what do you want to know?\n')
  output = process(text,question)
  if output:
    print(output)
  elif question == 'quit':
    break
  else:
    print("I don't know!")
  