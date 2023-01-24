import streamlit as st # For the UI
import pickle  # For loading the model

# For natural language processing related tasks
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


with open('models/vectorizer.h5','rb') as f:
    tfidf = pickle.load(f)

with open('models/spam_classifier_model.h5','rb') as f:
    model = pickle.load(f)


def transform_text(message):
    
    # Changing each word into lower case
    message = message.lower()
    
    # Creating a token list
    tokens = []
    message = nltk.word_tokenize(message)
    
    # Creating the stemmer object
    ps = PorterStemmer()  
    
    # Itterating through all the message tokens
    for word in message:
        
        # Removing all the non-alphanumeric and stop words
        if word.isalnum() and word not in stopwords.words('english'):
            
            # Stemming all the words(example: changing 'words' --> 'word')
            tokens.append(ps.stem(word))
    
    # Returning a string of cleaned tokens
    return ' '.join(tokens)

st.title('Email/SMS Messages Spam Classifier')

input_message = st.text_area('Enter the message here:')

if st.button('Check if it is spam'):

    # Preprocessing
    transformed_message = transform_text(input_message)

    # Vectorizing
    vectorized_input = tfidf.transform([transformed_message])

    # Predicting
    result = model.predict(vectorized_input)[0]

    # Displaying the result
    if result == 1:
        st.header('Spam')
    else:
        st.header('Not Spam')