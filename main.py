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
text = 'Originally, vegetables were collected from the wild by hunter-gatherers. Vegetables are all plants. Vegetables can be eaten either raw or cooked.'
question = 'What are vegetables?' 
sentence_tokens = nltk.sent_tokenize(text)
sentence_tokens.append(question)
sentence_tokens
tv = TfidfVectorizer(tokenizer=lemma_me)
tf=tv.fit_transform(sentence_tokens)
tf.toarray()
values = cosine_similarity(tf[-1], tf)
index = values.argsort()[0][-2]
values_flat = values.flatten()
values_flat.sort()
coeff = values_flat[-2]
if coeff>0.3:
    print(sentence_tokens[index])