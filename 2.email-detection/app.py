#author = 'ml_ypj'
#date:2022-08-23 23:13
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)
tfidf = pickle.load(open(r'C:\Users\DELL\data-science-learning\NLP\vectorizer.pkl','rb'))
model = pickle.load(open(r'C:\Users\DELL\data-science-learning\NLP\model.pkl','rb'))
st.title("垃圾短信/邮件分类器")
input_sms = st.text_area("输入你要检测的内容")
if st.button('Predict'):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("垃圾短信！")
    else:
        st.header("正常短信~")
