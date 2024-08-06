import streamlit as st
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.exceptions import NotFittedError
import joblib

# Load model and tokenizer
def load_model_and_tokenizer():
    try:
        with open("C:\\Users\\Edward Ofosu Mensah\\Downloads\\best_model (23).pkl", 'rb') as model_file:
            model = joblib.load(model_file)
        with open("C:\\Users\\Edward Ofosu Mensah\\Downloads\\tokenizer (2).pkl", 'rb') as tokenizer_file:
            tokenizer = pickle.load(tokenizer_file)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model or tokenizer: {e}")
        return None, None

model, tokenizer = load_model_and_tokenizer()

# Placeholder for course data
courses = [
    {"title": "Machine Learning for Everybody - Full Course", "thumbnail": "https://i.ytimg.com/vi/i_LwzRVP7bg/hq720.jpg?sqp=-oaymwEcCNAFEJQDSFXyq4qpAw4IARUAAIhCGAFwAcABBg==&rs=AOn4CLBP2UV9ostI2w6OAfjVSUqiIHi3YQ", "link": "https://www.youtube.com/watch?v=i_LwzRVP7bg"},
    {"title": "Machine Learning Engineer (Complete Roadmap)", "thumbnail": "https://i.ytimg.com/vi/7IgVGSaQPaw/hq720.jpg?sqp=-oaymwEcCNAFEJQDSFXyq4qpAw4IARUAAIhCGAFwAcABBg==&rs=AOn4CLAiUoQc0CfWTP0h1QhsbMVRFma6iA", "link": "https://www.youtube.com/watch?v=7IgVGSaQPaw"},
    {"title": "Machine Learning in 2024 - Beginners Course", "thumbnail": "https://i.ytimg.com/vi/bmmQA8A-yUA/hq720.jpg?sqp=-oaymwEcCNAFEJQDSFXyq4qpAw4IARUAAIhCGAFwAcABBg==&rs=AOn4CLCD6QVuw1XuhFVifRWMW2PoSm6J6Q", "link": "https://www.youtube.com/watch?v=bmmQA8A-yUA"},
    {"title": "Machine Learning Course for Beginners", "thumbnail": "https://i.ytimg.com/vi/NWONeJKn6kc/hq720.jpg?sqp=-oaymwEcCNAFEJQDSFXyq4qpAw4IARUAAIhCGAFwAcABBg==&rs=AOn4CLDUXO_HqgYZLXwn3TeWSBafYYqyug", "link": "https://www.youtube.com/watch?v=NWONeJKn6kc&t=1s"},
    {"title": "Python for Data Science Course", "thumbnail": "https://i.ytimg.com/vi/FTpmwX94_Yo/hq720.jpg?sqp=-oaymwEcCNAFEJQDSFXyq4qpAw4IARUAAIhCGAFwAcABBg==&rs=AOn4CLA8sjVRIJ-Eg-AaOIdrzhSwb3Obkw", "link": "https://www.youtube.com/watch?v=FTpmwX94_Yo&t=6882s"},
    {"title": "Understanding AI from Scratch", "thumbnail": "https://i.ytimg.com/vi/VgzHT9quo5c/hq720.jpg?sqp=-oaymwEcCNAFEJQDSFXyq4qpAw4IARUAAIhCGAFwAcABBg==&rs=AOn4CLDCU11hS3uH62CbwUJFHY8jtluRaw", "link": "https://www.youtube.com/watch?v=VgzHT9quo5c&t=10s"},
    
]

# Streamlit app title
st.sidebar.title("ML Hub")
st.sidebar.subheader("Navigation")

# Navigation
nav_option = st.sidebar.radio("Go to", ["Home", "Explore Our Courses", "Contact Us"])

# Home section
if nav_option == "Home":
    st.title("Welcome to the ML Learning hub")
    st.image("https://emeritus.org/in/wp-content/uploads/sites/3/2023/01/What-is-machine-learning-Definition-types-768x386.jpg.webp", width=800)
    st.write("Use the sidebar to navigate through the platform. You can view and submit reviews for different courses.")

# Reviews data
@st.cache_data
def load_data():
    try:
        return pd.read_csv("C:\\Users\\Edward Ofosu Mensah\\OneDrive - Ashesi University\\Desktop\\CompleteAIFinal\\reviews_extended.csv")
    except Exception as e:
        st.error(f"Error loading reviews data: {e}")
        return pd.DataFrame(columns=["User", "Review", "Rating"])

reviews_data = load_data()

# Courses and Reviews section
if nav_option == "Explore Our Courses":
    st.title("Dive In!")

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
            if not reviews_data.empty:
                for index, row in reviews_data.iterrows():
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
                    tokenizer.fit_on_texts([user_review])
                    review_vec = tokenizer.texts_to_sequences([user_review])
                    review_vec = pad_sequences(review_vec, maxlen=200)

                    # Check if model is fitted before predicting
                    try:
                        predicted_rating = model.predict(review_vec)[0]
                        rounded_rating = np.round(predicted_rating).astype(int)
                        star_rating = "★" * rounded_rating + "☆" * (5 - rounded_rating)
                        st.markdown(f"Predicted Star Rating: <span style='color: green;'>{star_rating}</span>", unsafe_allow_html=True)

                        # Save the review with predicted rating
                        new_review = pd.DataFrame({
                            "User": [user_name],
                            "Review": [user_review],
                            "Rating": [rounded_rating]
                        })

                        reviews_data = pd.concat([reviews_data, new_review], ignore_index=True)
                        reviews_data.to_csv("C:\\Users\\Edward Ofosu Mensah\\OneDrive - Ashesi University\\Desktop\\CompleteAIFinal\\reviews_extended.csv", index=False)

                        st.write("Review submitted successfully!")
                    except NotFittedError as e:
                        st.error(f"Model is not fitted: {e}")
                    except Exception as e:
                        st.error(f"Error in predicting rating: {e}")
                else:
                    st.write("Please enter both your name and review.")

# Contact Us section
if nav_option == "Contact Us":
    st.title("Contact Us")
    st.write("If you have any questions or need further assistance, please reach out to us:")
    st.write("**Email:** eddiemens0462@gmail.com / nanaafia@gmail.com")
    st.write("**Phone:** +123 456 7890")
    st.write("**Address:** 123 Learning Street, ML Hub City, Education Country")
    st.image("https://image.shutterstock.com/image-photo/contact-us-customer-support-hotline-260nw-1018999982.jpg", width=600)
