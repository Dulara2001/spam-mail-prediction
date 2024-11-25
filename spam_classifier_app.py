import streamlit as st
from joblib import load

# Load the trained model and TfidfVectorizer
model = load('logistic_regression_model.joblib')
vectorizer = load('tfidf_vectorizer.joblib')

# Streamlit app title
st.title("Spam Mail Predictor")

# Input field for the message
user_input = st.text_area("Enter your email message:")

# Button for prediction
if st.button("Predict"):
    if user_input.strip():  # Ensure input is not empty
        # Transform the input using the loaded TfidfVectorizer
        user_input_features = vectorizer.transform([user_input])

        # Make the prediction
        prediction = model.predict(user_input_features)

        # Determine the result
        if prediction[0] == 0:
            result = "Spam"
            background_color = "red"
        else:
            result = "Not Spam"
            background_color = "green"

        # Display the result with dynamic color
        st.markdown(
            f"""
            <div style='background-color: {background_color}; color: white; padding: 10px; border-radius: 5px; text-align: center;'>
                The message is classified as: <strong>{result}</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.error("Please enter a message to predict.")
