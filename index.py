import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
import plotly.express as px
import hashlib
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report
import io
from wordcloud import WordCloud, STOPWORDS  # NEW for Word Cloud

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')
vader = SentimentIntensityAnalyzer()

# ------------------ Sentiment Analysis Functions ------------------
def analyze_sentiment_vader(text):
    score = vader.polarity_scores(text)['compound']
    return "Positive" if score >= 0 else "Negative"

def analyze_sentiment_textblob(text):
    polarity = TextBlob(text).sentiment.polarity
    return "Positive" if polarity >= 0 else "Negative"

# ------------------ Password Hashing ------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ------------------ Session State Initialization ------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_type" not in st.session_state:
    st.session_state.user_type = None
if "page" not in st.session_state:
    st.session_state.page = "home"
if "admin_review_display" not in st.session_state:
    st.session_state.admin_review_display = False
if "registered_users" not in st.session_state:
    st.session_state.registered_users = {}

# ------------------ Data Loading ------------------
@st.cache_data
def load_data():
    file_path = 'Dataset.csv'
    return pd.read_csv(file_path) if os.path.exists(file_path) else None

df_original = load_data()

# ------------------ Helper Function for Navigation ------------------
def change_page(new_page):
    st.session_state.page = new_page
    if hasattr(st, 'experimental_rerun'):
        st.experimental_rerun()

# ------------------ Custom CSS for Styling ------------------
def set_background():
    page_bg = '''
    <style>
    body, .stApp {
        font-size: 18px;
    }
    .stApp {
        background: linear-gradient(45deg, #E3F2FD, #E8EAF6);
    }
    .stButton > button {
        background-color: #607D8B;
        color: white;
        font-size: 14px;
        border-radius: 8px;
        padding: 0.4em 0.8em;
        transition: background-color 0.3s ease;
        border: none;
    }
    .stButton > button:hover {
        background-color: #455A64;
        color: white;
    }
    .output-box {
        font-size: 18px;
        background-color: #f0f8ff;
        padding: 10px;
        border-radius: 8px;
        margin: 10px 0;
    }
    h1, h2, h3 {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #4B0082;
    }
    </style>
    '''
    st.markdown(page_bg, unsafe_allow_html=True)

set_background()

# ------------------ Page Functions ------------------

# Home Page
def home_page():
    st.title("🛍️ Fake Product Review Detection System")
    st.markdown("<h3 style='color:#4B0082;'>Detecting Fake Reviews, One Click at a Time</h3>", unsafe_allow_html=True)
    st.markdown("**Welcome!** Choose an option below:")
    col1, col2 = st.columns(2)
    with col1:
        st.button("Login", on_click=lambda: change_page("unified_login"))
    with col2:
        st.button("Register", on_click=lambda: change_page("register"))

# Single Login Page (for Admin, Registered, or Guest)
def unified_login_page():
    st.title("🔑 Unified Login")
    st.markdown("Enter your credentials. If you leave them blank, you'll proceed as a Guest User.")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Admin credentials (example)
    ADMIN_CREDENTIALS = {"admin1": "adminpass1", "admin2": "adminpass2"}

    def unified_login():
        # 1) Blank => Guest user
        if username.strip() == "" and password.strip() == "":
            st.session_state.authenticated = False
            st.session_state.user_type = "guest"
            change_page("guest_user")
            return

        # 2) Admin
        if username in ADMIN_CREDENTIALS and ADMIN_CREDENTIALS[username] == password:
            st.session_state.authenticated = True
            st.session_state.user_type = "admin"
            change_page("admin_dashboard")
            return

        # 3) Registered
        if username in st.session_state.registered_users and st.session_state.registered_users[username] == hash_password(password):
            st.session_state.authenticated = True
            st.session_state.user_type = "registered"
            change_page("registered_user")
            return

        # 4) Otherwise, invalid
        st.error("Invalid credentials. Try again or leave blank for Guest User.")

    st.button("Login", on_click=unified_login)
    st.button("Back", on_click=lambda: change_page("home"))

# Registration Page
def register_page():
    st.title("📝 User Registration")
    username = st.text_input("Enter Username")
    password = st.text_input("Enter Password", type="password")
    if st.button("Register"):
        if username in st.session_state.registered_users:
            st.error("Username already exists.")
        elif username == "" or password == "":
            st.error("Username and password cannot be empty.")
        else:
            st.session_state.registered_users[username] = hash_password(password)
            st.success("User Registered Successfully!")
            change_page("home")
    st.button("Back", on_click=lambda: change_page("home"))

# Registered User Dashboard
def registered_user_page():
    st.title("🔐 Registered User Dashboard")
    st.markdown("Welcome! Please choose an option:")
    st.button("Dataset Review", on_click=lambda: change_page("dataset_review"))
    st.button("Own Review", on_click=lambda: change_page("own_review"))
    st.button("Logout", on_click=lambda: change_page("home"))

# Guest User Page
def guest_user_page():
    st.title("🌟 Guest User Access")
    if df_original is None:
        st.warning("No dataset available.")
        st.button("Back", on_click=lambda: change_page("home"))
        return

    tab1, tab2, tab3 = st.tabs(["Dataset", "Charts", "Word Cloud"])

    # 1) Dataset
    with tab1:
        st.subheader("Dataset Display")
        st.markdown("<div class='output-box'><strong>Original Dataset</strong></div>", unsafe_allow_html=True)
        st.dataframe(df_original)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Cleaned Dataset")
        df_clean = df_original.dropna()
        st.dataframe(df_clean)

        analyzer_choice = st.radio("Select Sentiment Analyzer", ["VADER", "TextBlob"], index=0)
        if analyzer_choice == "VADER":
            df_clean['Predicted'] = df_clean['Text'].apply(lambda x: analyze_sentiment_vader(str(x)))
        else:
            df_clean['Predicted'] = df_clean['Text'].apply(lambda x: analyze_sentiment_textblob(str(x)))

        if 'Class' in df_clean.columns:
            try:
                acc = accuracy_score(df_clean['Class'], df_clean['Predicted'])
                report = classification_report(df_clean['Class'], df_clean['Predicted'], output_dict=True)
                st.markdown(f"<div class='output-box'><strong>Accuracy:</strong> {acc:.2f}</div>", unsafe_allow_html=True)
                st.markdown("<div class='output-box'><strong>Classification Report:</strong></div>", unsafe_allow_html=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
                csv_report = report_df.to_csv().encode('utf-8')
                st.download_button("Download Report", data=csv_report, file_name="classification_report.csv", mime="text/csv")
            except Exception as e:
                st.error("Error computing metrics. Ensure the 'Class' column has appropriate labels.")
        else:
            st.info("No 'Class' column found for evaluation.")
        st.button("Back", on_click=lambda: change_page("home"))

    # 2) Charts
    with tab2:
        st.subheader("Interactive Charts")
        # Class Distribution
        st.markdown("<div class='output-box'><strong>Class Distribution</strong></div>", unsafe_allow_html=True)
        if 'Class' in df_original.columns:
            class_counts = df_original['Class'].value_counts().reset_index()
            class_counts.columns = ['Class', 'Count']
            fig1 = px.bar(class_counts, x='Class', y='Count', title="Class Distribution", color='Class')
            st.plotly_chart(fig1)
        else:
            st.info("No 'Class' column available.")

        # Polarity Distribution
        st.markdown("<div class='output-box'><strong>Sentiment Polarity Distribution</strong></div>", unsafe_allow_html=True)
        df_original['Polarity'] = df_original['Text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        fig2 = px.histogram(df_original, x='Polarity', nbins=20, title="Polarity Distribution", color_discrete_sequence=["blue"])
        st.plotly_chart(fig2)

        # Pie Chart
        st.markdown("<div class='output-box'><strong>Review Sentiment Pie Chart</strong></div>", unsafe_allow_html=True)
        polarity_series = df_original['Text'].apply(lambda x: "Positive" if TextBlob(str(x)).sentiment.polarity >= 0 else "Negative")
        sentiment_counts = polarity_series.value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']
        fig3 = px.pie(sentiment_counts, values='Count', names='Sentiment', title="Sentiment Distribution", color_discrete_map={'Positive':'#66b3ff','Negative':'#ff9999'})
        st.plotly_chart(fig3)

        # Predicted Sentiment Distribution
        st.markdown("<div class='output-box'><strong>Predicted Sentiment Distribution</strong></div>", unsafe_allow_html=True)
        df_temp = df_original.copy()
        df_temp['Predicted'] = df_temp['Text'].apply(lambda x: analyze_sentiment_textblob(str(x)))
        pred_counts = df_temp['Predicted'].value_counts().reset_index()
        pred_counts.columns = ['Sentiment', 'Count']
        fig4 = px.bar(pred_counts, x='Sentiment', y='Count', title="Predicted Sentiment Distribution", color='Sentiment')
        st.plotly_chart(fig4)

    # 3) Word Cloud
    with tab3:
        st.subheader("Word Cloud of Reviews")
        # Combine all text
        all_text = " ".join(str(txt) for txt in df_original['Text'].dropna())
        # Generate Word Cloud
        wordcloud = WordCloud(stopwords=STOPWORDS, background_color="white", max_words=1000).generate(all_text)
        plt.figure(figsize=(10, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        st.pyplot(plt.gcf())
        st.info("Visual representation of the most frequent words in the reviews.")

# Dataset Review Page (Registered Users)
def dataset_review():
    st.title("📊 Dataset Review")
    if df_original is not None:
        index = st.number_input("Enter Index:", min_value=0, max_value=len(df_original)-1, step=1, key="dataset_index")
        if st.button("Review"):
            st.markdown(f"<div class='output-box'><strong>Selected Review:</strong> {df_original.iloc[index]['Text']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='output-box'><strong>Name:</strong> {df_original.iloc[index]['Name']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='output-box'><strong>Class:</strong> {df_original.iloc[index]['Class']}</div>", unsafe_allow_html=True)
    else:
        st.warning("Dataset not loaded!")
    st.button("Back", on_click=lambda: change_page("registered_user"))

# Own Review Analysis Page (Registered Users)
def own_review():
    st.title("📢 Own Review Analysis")
    user_review = st.text_area("Enter your review")
    analyzer_choice = st.radio("Select Sentiment Analyzer", ["VADER", "TextBlob"], index=0)
    if st.button("Analyze Review"):
        if analyzer_choice == "VADER":
            result = analyze_sentiment_vader(user_review)
        else:
            result = analyze_sentiment_textblob(user_review)
        st.markdown(f"<div class='output-box'><strong>Sentiment:</strong> {result}</div>", unsafe_allow_html=True)
    st.button("Back", on_click=lambda: change_page("registered_user"))

# Admin Dashboard Page (Unchanged)
def admin_dashboard():
    st.title("📊 Admin Dashboard")
    if df_original is not None:
        index = st.number_input("Enter Index:", min_value=0, max_value=len(df_original)-1, step=1, key="admin_index")
        if st.button("Show Review"):
            st.session_state.admin_review_display = True
            st.session_state.selected_index = index

        if st.session_state.get("admin_review_display", False):
            selected_index = st.session_state.selected_index
            st.markdown(f"<div class='output-box'><strong>Selected Review:</strong> {df_original.iloc[selected_index]['Text']}</div>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                show_age_name_gender = st.checkbox("Show Age, Name, Gender", key="show_age_name_gender")
                show_class = st.checkbox("Show Class", key="show_class")
            with col2:
                show_polarity = st.checkbox("Show Polarity", key="show_polarity")
                show_rating = st.checkbox("Show Rating", key="show_rating")
                show_product = st.checkbox("Show Product Link", key="show_product")
            if show_age_name_gender:
                st.markdown("<div class='output-box'><strong>User Information:</strong></div>", unsafe_allow_html=True)
                st.dataframe(df_original.iloc[[selected_index]][["Age", "Gender", "Name"]])
            if show_class:
                st.markdown(f"<div class='output-box'><strong>Class:</strong> {df_original.iloc[selected_index]['Class']}</div>", unsafe_allow_html=True)
            if show_polarity:
                polarity = TextBlob(str(df_original.iloc[selected_index]['Text'])).sentiment.polarity
                st.markdown(f"<div class='output-box'><strong>Polarity:</strong> {polarity:.2f}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='output-box'><strong>Prediction:</strong> {df_original.iloc[selected_index]['Polarity']}</div>", unsafe_allow_html=True)
            if show_rating:
                st.markdown(f"<div class='output-box'><strong>Rating:</strong> {df_original.iloc[selected_index]['Rating']}</div>", unsafe_allow_html=True)
            if show_product:
                st.markdown(f"<div class='output-box'><strong>Product Link:</strong> {df_original.iloc[selected_index]['Product Link']}</div>", unsafe_allow_html=True)
    else:
        st.warning("Dataset not loaded!")
    st.button("Logout", on_click=lambda: change_page("home"))

# ------------------ Page Routing ------------------
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "unified_login":
    unified_login_page()
elif st.session_state.page == "register":
    register_page()
elif st.session_state.page == "registered_user":
    registered_user_page()
elif st.session_state.page == "guest_user":
    guest_user_page()
elif st.session_state.page == "dataset_review":
    dataset_review()
elif st.session_state.page == "own_review":
    own_review()
elif st.session_state.page == "admin_dashboard":
    admin_dashboard()
