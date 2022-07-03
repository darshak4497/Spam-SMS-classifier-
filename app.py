import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

def transform_text(text):
  text=text.lower()

  text=nltk.word_tokenize(text)

  y=[]
  for i in text:
    if i.isalnum():
      y.append(i)

  text=y

  a=[]
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      a.append(i)
  b=[]
  for i in a:
    b.append(ps.stem(i))


  return " ".join(b)

tfidf=pickle.load(open('vectoriser.pkl', 'rb'), encoding='UTF-8')
model=pickle.load(open('model.pkl', 'rb'), encoding='UTF-8')

st.title('email/sms spam classifier')

input_sms=st.text_input("Enter the message")

#1.preprocess the inputstre text
transformed_sms=transform_text(input_sms)
#2. vectorize
vector_input=tfidf.transform(['transformed_sms'])
#3.predict
result=model.predict(vector_input)[0]
#4. display
if result==1:
  st.header('spam')

else:
  st.header('not spam')


