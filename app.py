import streamlit as st
import pickle
import string
from nltk.corpus import stopwords

# Define the text_process function
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)

    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]

# Load the pre-trained model
with open('pipeline.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

st.title('Spam/Ham Classification')

# Text input from the user
user_input = st.text_area("Enter the message you want to classify:", "")

if st.button("Classify"):
    if user_input:
        # Predict the class of the input message
        prediction = model.predict([user_input])
        prediction_proba = model.predict_proba([user_input])
        
        # Display the result
        if prediction[0] == 'spam':
            st.error(f'This message is classified as SPAM.')
        else:
            st.success(f'This message is classified as HAM.')

        # Display the prediction probabilities
        st.write("Prediction probabilities:")
        st.write(f"Spam: {prediction_proba[0][model.classes_ == 'spam'][0]:.2f}")
        st.write(f"Ham: {prediction_proba[0][model.classes_ == 'ham'][0]:.2f}")
    else:
        st.warning("Please enter a message to classify.")
