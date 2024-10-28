


import os
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as gen_ai
import firebase_admin
from firebase_admin import credentials, auth, firestore
from streamlit_option_menu import option_menu
import pyrebase
import json
import matplotlib.pyplot as plt
import numpy as np


# Load environment variables
load_dotenv()

# Initialize Firebase Admin SDK only once
if not firebase_admin._apps:
    cred = credentials.Certificate('firebaseconfig.json')
    firebase_admin.initialize_app(cred)

# Firestore Database
db = firestore.client()

# Firebase auth
with open('firebaseconfig.json') as f:
    firebase_config = json.load(f)

firebase = pyrebase.initialize_app(firebase_config)
auth = firebase.auth()

# Configure Streamlit page settings
st.set_page_config(
    page_title="Chat with Gemini-Pro!",
    page_icon=":brain:",  # Favicon emoji
    layout="centered",  # Page layout option
)



# Function for user login and signup
import streamlit as st
from firebase_admin import firestore  # Ensure Firebase is initialized and Firestore is imported
from streamlit_option_menu import option_menu

def login_signup():
    choice = st.selectbox("Login/Signup", ["Login", "Signup"])
    email = st.text_input("Enter your email")
    password = st.text_input("Enter your password", type="password")

    # Signup Logic
    if choice == "Signup" and st.button("Create Account"):
        if email and password:
            try:
                user = auth.create_user_with_email_and_password(email=email, password=password)
                st.success("Account created successfully!")
            except Exception as e:
                st.error(f"Error creating account: {e}")
                print(e)  # Debug statement
        else:
            st.error("Please enter both email and password.")

    # Login Logic
    if choice == "Login" and st.button("Login"):
        if email and password:
            try:
                # Use Firebase Authentication API to sign in
                user = auth.sign_in_with_email_and_password(email, password)
                st.session_state.user = user
                st.session_state.logged_in = True  # Set login flag to True
                st.session_state.user_id = user['localId']  # Store user ID for later use
                st.success("Logged in successfully!")
                st.rerun()  # Rerun the script to show the main app
            except Exception as e:
                st.error(f"Invalid credentials: {e}")
                print(e)  # Debug statement
        else:
            st.error("Please enter both email and password.")

# If user is not logged in, show login/signup page
if 'logged_in' not in st.session_state or not st.session_state.logged_in:
    login_signup()
else:
    user_id = st.session_state.user_id  # Ensure user ID is available for later use

    # Sidebar Navigation
    with st.sidebar:
        selected = option_menu(
            menu_title="Menu",
            options=["Chatbot", "Prediction", "Profile"],
            icons=["house-heart-fill", "calendar2-heart-fill", "envelope-heart-fill"],
            menu_icon="heart-eyes-fill",
            default_index=0,
        )

    # Firestore Profile Retrieval (with cache clearing for updates)
    @st.cache_data
    def get_profile_data(user_id):
        profile_doc = db.collection("users").document(user_id).get()
        if profile_doc.exists:
            return profile_doc.to_dict()
        return {}  # Return empty dictionary if no data exists

    # Profile Section
    if selected == "Profile":
        col1, col2 = st.columns([7, 1])

        with col1:
            st.subheader("Build Your Profile")
        with col2:
            if st.button("Logout"):
                del st.session_state.user
                del st.session_state.logged_in
                st.rerun()

        # Load existing profile data from Firestore
        profile_data = get_profile_data(user_id)

        # Pre-fill the form with existing data or defaults
        name = st.text_input("Name", value=profile_data.get("name", ""))
        age = st.text_input("Age", value=str(profile_data.get("age", "")))  # Convert to str
        months_pregnant = st.text_input("Months Pregnant", value=str(profile_data.get("months_pregnant", "")))  # Convert to str
        chronic_diseases = st.text_area("Chronic Diseases (separate by commas)", 
                                        value=', '.join(profile_data.get("chronic_diseases", [])))
        weight = st.text_input("Weight (kg)", value=str(profile_data.get("weight", "")))  # Convert to str
        height = st.text_input("Height (cm)", value=str(profile_data.get("height", "")))  # Convert to str
        medications = st.text_area("Medications", value=', '.join(profile_data.get("medications", [])))
        allergies = st.text_area("Allergies", value=', '.join(profile_data.get("allergies", [])))
        exercise = st.text_input("Exercise Routine", value=profile_data.get("exercise", ""))
        dietary_preferences = st.text_input("Dietary Preferences", value=profile_data.get("dietary_preferences", ""))
        smoking_habits = st.text_input("Smoking Habits", value=profile_data.get("smoking_habits", ""))
        alcohol_habits = st.text_input("Alcohol Consumption", value=profile_data.get("alcohol_habits", ""))

        # Save profile data to Firestore on button click
        if st.button("Save Profile"):
            # Update profile data and reformat lists correctly
            updated_profile_data = {
                "name": name,
                "age": int(age) if age.isdigit() else age,
                "months_pregnant": int(months_pregnant) if months_pregnant.isdigit() else months_pregnant,
                "chronic_diseases": [disease.strip() for disease in chronic_diseases.split(',') if disease.strip()],
                "weight": float(weight) if weight.replace('.', '', 1).isdigit() else weight,
                "height": float(height) if height.replace('.', '', 1).isdigit() else height,
                "medications": [med.strip() for med in medications.split(',') if med.strip()],
                "allergies": [allergy.strip() for allergy in allergies.split(',') if allergy.strip()],
                "exercise": exercise,
                "dietary_preferences": dietary_preferences,
                "smoking_habits": smoking_habits,
                "alcohol_habits": alcohol_habits
            }

            # Save to Firestore
            db.collection("users").document(user_id).set(updated_profile_data)
            st.success("Profile saved successfully!")
            print(f"Updated Profile Data for {user_id}: {updated_profile_data}")

            # Clear the cache to fetch updated data on next load
            get_profile_data.clear()


    # Chatbot Section
    # Chatbot Section
    elif selected == "Chatbot":

        # Load API key from environment variables
        GOOGLE_API_KEY = os.getenv("API_Key")

        # Set up Google Gemini-Pro AI model
        gen_ai.configure(api_key=GOOGLE_API_KEY)
        gen_ai.configure(api_key=GOOGLE_API_KEY)
        model = gen_ai.GenerativeModel('gemini-pro')

        # Function to translate roles between Gemini-Pro and Streamlit terminology
        def translate_role_for_streamlit(user_role):
            return "assistant" if user_role == "model" else user_role

        # Initialize chat session and display history in Streamlit
        if "chat_session" not in st.session_state:
            st.session_state.chat_session = model.start_chat(history=[])
            st.session_state.display_history = []  # New: Store only user prompt and assistant response here
        
        if "display_history" not in st.session_state:
            st.session_state.display_history = []  # Stores only user prompts and assistant responses


        # Function to retrieve profile data (assuming get_profile_data function exists)
        def get_profile_data(user_id):
            # Mock profile data for the example
            return {
                "name": "Aayushi",
                "age": 21,
                "months_pregnant": 4,
                "chronic_diseases": ["Diabetes", "Asthma"],
                "weight": 34.0,
                "height": 164.0,
                "medications": ["diabetes", "asthma"],
                "allergies": ["peanuts"],
                "exercise": "walking",
                "dietary_preferences": "vegetarian",
                "smoking_habits": "no",
                "alcohol_habits": "no"
            }

        # Fetch profile data
        profile_data = get_profile_data(user_id)

        # Function to get profile data as a formatted string
        def get_profile_info():
            return "\n".join([
                f"Name: {profile_data.get('name', 'N/A')}",
                f"Age: {profile_data.get('age', 'N/A')}",
                f"Months Pregnant: {profile_data.get('months_pregnant', 'N/A')}",
                f"Chronic Diseases: {', '.join(profile_data.get('chronic_diseases', []))}",
                f"Weight: {profile_data.get('weight', 'N/A')} kg",
                f"Height: {profile_data.get('height', 'N/A')} cm",
                f"Medications: {', '.join(profile_data.get('medications', []))}",
                f"Allergies: {', '.join(profile_data.get('allergies', []))}",
                f"Exercise Routine: {profile_data.get('exercise', 'N/A')}",
                f"Dietary Preferences: {profile_data.get('dietary_preferences', 'N/A')}",
                f"Smoking Habits: {profile_data.get('smoking_habits', 'N/A')}",
                f"Alcohol Consumption: {profile_data.get('alcohol_habits', 'N/A')}"
            ])

        # Chatbot UI
        st.title("🤖 Your AI Health Assistant")
        for message in st.session_state.display_history:
            if message["role"] == "user":
                st.chat_message("user").markdown(message["text"])  # Display user prompt
            else:
                st.chat_message("assistant").markdown(message["text"])  # Display assistant response

        # Get user input
        user_prompt = st.chat_input("Ask Gemini-Pro...")
        if user_prompt:
            # Prepare full prompt with profile data for the model
            complete_prompt = f"{get_profile_info()}\n\nUser Query: {user_prompt}"
            
            # Display user's question in the chat
            with st.chat_message("user"):
                st.markdown(user_prompt)
            
            # Send prompt to Gemini-Pro
            response = st.session_state.chat_session.send_message(complete_prompt)
            
            # Store user prompt and response for display only
            st.session_state.display_history.append({"role": "user", "text": user_prompt})
            st.session_state.display_history.append({"role": "assistant", "text": response.text})
            
            # Display assistant’s response
            with st.chat_message("assistant"):
                st.markdown(response.text)

    
    elif selected == "Prediction":

        def predict_risk(age, systolic_bp, diastolic_bp, bs, heart_rate):
            # Placeholder for your ML model prediction logic
            # This function should return a risk level based on input parameters
            # For demonstration, we use a simple calculation
            risk_level = (age + systolic_bp + diastolic_bp + bs + heart_rate) / 10
            return risk_level

        def plot_graph(risk_level):
            # Create a simple bar chart based on the risk level
            categories = ['Risk Level']
            values = [risk_level]
            
            fig, ax = plt.subplots()
            ax.bar(categories, values, color='blue')
            ax.set_ylim(0, 100)  # Set limit for the risk level
            ax.set_ylabel('Risk Intensity Level')
            ax.set_title('Predicted Risk Intensity Level')
            
            return fig

        # Page Title
        st.title("Prediction")

        # Input Section
        st.header("Input Patient Data")
    
        age = st.slider("Age:", min_value=18, max_value=50, value=30, step=1)
        systolic_bp = st.slider("Systolic BP (mmHg):", min_value=90, max_value=200, value=120, step=1)
        diastolic_bp = st.slider("Diastolic BP (mmHg):", min_value=60, max_value=120, value=80, step=1)
        bs = st.slider("Blood Glucose Levels (mmol/L):", min_value=3.0, max_value=15.0, value=5.0, step=0.1)
        heart_rate = st.slider("Heart Rate (bpm):", min_value=40, max_value=150, value=70, step=1)

        # Button to Trigger Prediction
        if st.button("Predict Risk Level"):
            risk_level = predict_risk(age, systolic_bp, diastolic_bp, bs, heart_rate)
            
            # Displaying the Risk Level
            st.success(f"Predicted Risk Level: {risk_level:.2f}")
            
            # Plot the graph
            fig = plot_graph(risk_level)
            st.pyplot(fig)
