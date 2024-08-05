# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:51:59 2024

@author: Edward Ofosu Mensah
"""

import streamlit as st
import pickle
import pandas as pd
import random
import os

# Load model and tokenizer
def load_model_and_tokenizer():
    with open("bestmodel.pkl", 'rb') as model_file:
        model = pickle.load(model_file)
    with open("tokenizer.pkl", 'rb') as tokenizer_file:
        tokenizer = pickle.load(tokenizer_file)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

# Placeholder for course data
courses = [
    {"title": "Machine Learning for Everybody - Full Course", "thumbnail": "https://i.ytimg.com/vi/i_LwzRVP7bg/hq720.jpg?sqp=-oaymwEcCNAFEJQDSFXyq4qpAw4IARUAAIhCGAFwAcABBg==&rs=AOn4CLBP2UV9ostI2w6OAfjVSUqiIHi3YQ", "link": "https://www.youtube.com/watch?v=i_LwzRVP7bg"},
    {"title": "Machine Learning Engineer (Complete Roadmap)", "thumbnail": "https://i.ytimg.com/vi/7IgVGSaQPaw/hq720.jpg?sqp=-oaymwEcCNAFEJQDSFXyq4qpAw4IARUAAIhCGAFwAcABBg==&rs=AOn4CLAiUoQc0CfWTP0h1QhsbMVRFma6iA", "link": "https://www.youtube.com/watch?v=7IgVGSaQPaw"},
    {"title": "Machine Learning in 2024 - Beginners Course", "thumbnail": "https://i.ytimg.com/vi/bmmQA8A-yUA/hq720.jpg?sqp=-oaymwEcCNAFEJQDSFXyq4qpAw4IARUAAIhCGAFwAcABBg==&rs=AOn4CLCD6QVuw1XuhFVifRWMW2PoSm6J6Q", "link": "https://www.youtube.com/watch?v=bmmQA8A-yUA"}
]

# Streamlit app title
st.title("Course Review Platform")

# Reviews data
@st.cache_data
def load_data():
    return pd.read_csv("reviews.csv")

reviews_data = load_data()

# Initialize session state for review visibility
if 'show_reviews' not in st.session_state:
    st.session_state['show_reviews'] = {course["title"]: False for course in courses}

# Display courses and reviews
for course in courses:
    st.header(course["title"])
    st.image(course["thumbnail"], width=600)
    st.write(f"[Course Link]({course['link']})")

    # Button to toggle review visibility
    if st.button(f"Show Reviews for {course['title']}"):
        st.session_state['show_reviews'][course["title"]] = not st.session_state['show_reviews'][course["title"]]

    # Display or hide reviews based on session state
    if st.session_state['show_reviews'][course["title"]]:
        course_reviews = reviews_data[reviews_data['Course'] == course['title']]
        if not course_reviews.empty:
            for index, row in course_reviews.iterrows():
                stars = "★" * int(row['Rating']) + "☆" * (5 - int(row['Rating']))
                st.markdown(f"**{row['User']}**: {row['Review']}")
                st.markdown(f"<span style='color: green;'>{stars}</span>", unsafe_allow_html=True)
        else:
            st.write("No reviews available.")

    # Add a new review section for each course
    with st.expander(f"Add a Review for {course['title']}"):
        user_name = st.text_input(f"Your Name for {course['title']}")
        user_review = st.text_area(f"Your Review for {course['title']}")

        if st.button(f"Submit Review for {course['title']}"):
            if user_name and user_review:
                # Predict the rating for the review
                review_vec = tokenizer.texts_to_sequences([user_review])
                review_vec = pad_sequences(review_vec, maxlen=200)
                predicted_rating = model.predict(review_vec)[0][0]

                # Convert the predicted rating to stars and display in green
                star_rating = "★" * int(predicted_rating) + "☆" * (5 - int(predicted_rating))
                st.markdown(f"Predicted Star Rating: <span style='color: green;'>{star_rating}</span>", unsafe_allow_html=True)

                # Save the review with predicted rating
                new_review = pd.DataFrame({
                    "User": [user_name],
                    "Review": [user_review],
                    "Rating": [predicted_rating],
                    "Course": [course['title']]
                })

                reviews_data = pd.concat([reviews_data, new_review], ignore_index=True)
                reviews_data.to_csv("reviews.csv", index=False)

                st.write("Review submitted successfully!")
            else:
                st.write("Please enter both your name and review.")
