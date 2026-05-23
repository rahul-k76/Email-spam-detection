import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import streamlit as st
from pathlib import Path

# Load dataset using a relative, deployment-friendly path with uploader fallback
data_path = Path(__file__).resolve().parent / "spam.csv"
if data_path.exists():
    data = pd.read_csv(data_path)
else:
    try:
        data = pd.read_csv("D:\\MLProject\\SpamDetection\\spam.csv")
    except Exception:
        st.error(f"Dataset not found at {data_path}. Please upload 'spam.csv'.")
        upload = st.file_uploader("Upload spam.csv", type=["csv"])
        if upload is not None:
            data = pd.read_csv(upload)
        else:
            st.stop()

data.drop_duplicates(inplace=True)
data['Category'] = data['Category'].replace(['ham', 'spam'], ['This is Not Spam', 'This is Spam message'])

# Prepare training data
mess = data['Message']
cat = data['Category']
mess_train, mess_test, cat_train, cat_test = train_test_split(mess, cat, test_size=0.2, random_state=42)

# Vectorization
cv = CountVectorizer(stop_words='english')
features = cv.fit_transform(mess_train)

# Create and train model
model = MultinomialNB()
model.fit(features, cat_train)

#test score
# features_test = cv.transform(mess_test)
# print(model.score(features_test,cat_test))

# Define prediction function
def predict(message):
    input_message = cv.transform([message])  # No need to convert to array
    result = model.predict(input_message)[0]  # Extract first element to avoid NumPy array issues
    return result

# Streamlit UI
st.header('Spam Detection')
input_mess = st.text_input('Enter Message Here')
if st.button('SpamCheck'):
    output = predict(input_mess)
    st.markdown(f'**Prediction:** {output}')
