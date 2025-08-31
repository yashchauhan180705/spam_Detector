import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
import imaplib
import email
from email.header import decode_header
import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from gymnasium import spaces
import re
import ssl
import socket

# File paths
MODEL_PATH = "spam_dqn_model"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
SCALER_PATH = "scaler.pkl"
DATA_PATH = "enron_spam_data.csv"

# Test with these settings for Gmail:
Email: "cyash7420@gmail.com"
Password: "andvttbebjpskhvk"
IMAP_Server: "imap.gmail.com"
Port: 993

# Initialize session state
if 'emails_df' not in st.session_state:
    st.session_state.emails_df = pd.DataFrame()
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'model' not in st.session_state:
    st.session_state.model = None


# Custom Gym Environment
class SpamEnv(gym.Env):
    def __init__(self, X, y):
        super(SpamEnv, self).__init__()
        self.X = X.astype(np.float32)
        self.y = y
        self.current_index = 0
        self.max_steps = len(X)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(X.shape[1],), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_index = 0
        obs = self.X[self.current_index]
        return obs, {}

    def step(self, action):
        true_label = self.y[self.current_index]
        reward = 1.0 if action == true_label else -1.0
        self.current_index += 1
        terminated = self.current_index >= self.max_steps
        next_obs = self.X[self.current_index] if not terminated else np.zeros(self.X.shape[1], dtype=np.float32)
        return next_obs, reward, terminated, False, {}


def load_or_train_model():
    """Load existing model or train new one"""
    if st.session_state.model_loaded:
        return st.session_state.model, st.session_state.vectorizer, st.session_state.scaler

    try:
        # Check if we have a pre-trained model
        if os.path.exists(MODEL_PATH + ".zip") and os.path.exists(VECTORIZER_PATH) and os.path.exists(SCALER_PATH):
            model = DQN.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            scaler = joblib.load(SCALER_PATH)
            st.session_state.model = model
            st.session_state.vectorizer = vectorizer
            st.session_state.scaler = scaler
            st.session_state.model_loaded = True
            return model, vectorizer, scaler
        else:
            st.warning("Pre-trained model not found. Please train the model first with your dataset.")
            return None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None


def train_model_with_dataset():
    """Train the model with the dataset"""
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset file '{DATA_PATH}' not found. Please upload the dataset.")
        return False

    try:
        # Load dataset
        df = pd.read_csv(DATA_PATH)
        df.dropna(subset=['Message', 'Spam/Ham'], inplace=True)
        df['label'] = df['Spam/Ham'].map({'ham': 0, 'spam': 1})
        X_text = df['Message']
        y = df['label'].values

        # TF-IDF and Scaling
        vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        X = vectorizer.fit_transform(X_text).toarray()
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        # Save vectorizer and scaler
        joblib.dump(vectorizer, VECTORIZER_PATH)
        joblib.dump(scaler, SCALER_PATH)

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Create environment
        def make_env():
            return SpamEnv(X_train, y_train)

        env = make_vec_env(make_env, n_envs=1)

        # Create and train model
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0001,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=32,
            gamma=0.99,
            exploration_fraction=0.1,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.02,
            train_freq=4,
            target_update_interval=1000
        )

        # Train the model
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(5):  # Training in steps for progress visualization
            model.learn(total_timesteps=10000)
            progress_bar.progress((i + 1) / 5)
            status_text.text(f'Training progress: {((i + 1) / 5) * 100:.0f}%')

        # Save model
        model.save(MODEL_PATH)

        # Update session state
        st.session_state.model = model
        st.session_state.vectorizer = vectorizer
        st.session_state.scaler = scaler
        st.session_state.model_loaded = True

        st.success("Model trained successfully!")
        return True

    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return False


def clean_text(text):
    """Clean email text for better processing"""
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespaces
    text = ' '.join(text.split())
    return text


def decode_mime_words(s):
    """Decode MIME encoded words"""
    if s is None:
        return ""
    decoded_parts = []
    for part, encoding in decode_header(s):
        if isinstance(part, bytes):
            try:
                if encoding:
                    part = part.decode(encoding)
                else:
                    part = part.decode('utf-8', errors='ignore')
            except:
                part = part.decode('utf-8', errors='ignore')
        decoded_parts.append(str(part))
    return ''.join(decoded_parts)


def fetch_emails(email_address, password, imap_server, port=993, max_emails=50):
    """Fetch emails from IMAP server"""
    try:
        # Create SSL context with timeout
        context = ssl.create_default_context()

        # Connect to server with timeout
        socket.setdefaulttimeout(30)  # 30 seconds timeout
        mail = imaplib.IMAP4_SSL(imap_server, port, ssl_context=context)

        # Login with better error handling
        try:
            mail.login(email_address, password)
        except imaplib.IMAP4.error as login_error:
            error_msg = str(login_error)
            if "Application-specific password required" in error_msg or "ALERT" in error_msg:
                raise Exception(
                    "Gmail requires App-Specific Password. Please generate one from Google Account Settings → Security → App passwords")
            elif "authentication failed" in error_msg.lower():
                raise Exception("Authentication failed. Check your email and password/app-password")
            elif "invalid credentials" in error_msg.lower():
                raise Exception("Invalid credentials. For Gmail, use App-Specific Password instead of regular password")
            else:
                raise Exception(f"Login failed: {error_msg}")

        # Select inbox
        status, select_result = mail.select('inbox')
        if status != 'OK':
            raise Exception(f"Failed to select inbox: {select_result}")

        # Search for all emails
        status, messages = mail.search(None, 'ALL')

        if status != 'OK':
            raise Exception("Failed to search emails")

        # Get message IDs
        message_ids = messages[0].split()

        if not message_ids:
            raise Exception("No emails found in inbox")

        # Limit the number of emails
        message_ids = message_ids[-max_emails:]

        emails_data = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, msg_id in enumerate(message_ids):
            try:
                # Fetch email
                status, msg_data = mail.fetch(msg_id, '(RFC822)')

                if status != 'OK':
                    continue

                # Parse email
                email_body = msg_data[0][1]
                email_message = email.message_from_bytes(email_body)

                # Extract email details
                subject = decode_mime_words(email_message["Subject"]) or "No Subject"
                sender = decode_mime_words(email_message["From"]) or "Unknown Sender"
                receiver = decode_mime_words(email_message["To"]) or email_address
                date_str = email_message["Date"] or ""

                # Extract email content
                content = ""
                if email_message.is_multipart():
                    for part in email_message.walk():
                        if part.get_content_type() == "text/plain":
                            try:
                                content = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                                break
                            except:
                                continue
                else:
                    try:
                        content = email_message.get_payload(decode=True).decode('utf-8', errors='ignore')
                    except:
                        content = str(email_message.get_payload())

                # Clean content
                content = clean_text(content)

                emails_data.append({
                    'Subject': subject[:100] + "..." if len(subject) > 100 else subject,
                    'Sender': sender,
                    'Receiver': receiver,
                    'Date': date_str,
                    'Content': content[:500] + "..." if len(content) > 500 else content,
                    'Full_Content': content,
                    'Selected': False,
                    'Spam_Status': 'Not Checked'
                })

                # Update progress
                progress = (i + 1) / len(message_ids)
                progress_bar.progress(progress)
                status_text.text(f'Fetching emails: {i + 1}/{len(message_ids)}')

            except Exception as e:
                # Log individual email errors but continue
                continue

        # Close connection
        mail.close()
        mail.logout()

        progress_bar.empty()
        status_text.empty()

        if not emails_data:
            raise Exception("No emails could be processed. Check your email content or try with fewer emails.")

        return pd.DataFrame(emails_data)

    except Exception as e:
        st.error(f"Error fetching emails: {str(e)}")

        # Provide specific guidance based on error type
        if "App-Specific Password" in str(e):
            st.error("🔑 **Action Required**: Generate a Gmail App Password")
            st.markdown("""
            **Quick Steps:**
            1. Go to [Google Account Security](https://myaccount.google.com/security)
            2. Enable 2-Step Verification if not already enabled
            3. Click **App passwords**
            4. Generate password for **Mail**
            5. Use the generated 16-character password here
            """)
        elif "authentication failed" in str(e).lower():
            st.error("🔐 **Authentication Issue**: Check your credentials")
            if "gmail" in imap_server.lower():
                st.info("For Gmail: Use App-Specific Password, not your regular Google password")
        elif "timeout" in str(e).lower():
            st.error("⏱️ **Connection Timeout**: Try again or check your internet connection")

        return pd.DataFrame()


def predict_spam(content, model, vectorizer, scaler):
    """Predict if email content is spam"""
    try:
        if not content or pd.isna(content):
            return "Unknown"

        # Vectorize content
        content_vector = vectorizer.transform([str(content)]).toarray()
        content_vector = scaler.transform(content_vector)

        # Predict
        action, _ = model.predict(content_vector, deterministic=True)
        return "Spam" if action == 1 else "Ham"
    except:
        return "Error"


# Streamlit UI
st.set_page_config(page_title="📧 Advanced Spam Email Detector", layout="wide")

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    text-align: center;
    color: #1f77b4;
    font-size: 2.5rem;
    margin-bottom: 2rem;
}
.section-header {
    color: #2e7d32;
    font-size: 1.5rem;
    margin: 1.5rem 0 1rem 0;
    border-bottom: 2px solid #e8f5e8;
    padding-bottom: 0.5rem;
}
.email-card {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    margin: 1rem 0;
    border-left: 4px solid #1f77b4;
}
.spam-indicator {
    font-weight: bold;
    padding: 4px 8px;
    border-radius: 4px;
}
.spam {
    background-color: #ffebee;
    color: #c62828;
}
.ham {
    background-color: #e8f5e8;
    color: #2e7d32;
}
.not-checked {
    background-color: #fff3e0;
    color: #f57c00;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">📧 Advanced Spam Email Detector</div>', unsafe_allow_html=True)

# Sidebar for model management
st.sidebar.markdown("### 🔧 Model Management")

# Model training section
if st.sidebar.button("Train New Model"):
    with st.spinner("Training model... This may take a few minutes."):
        train_model_with_dataset()

# Load model
model, vectorizer, scaler = load_or_train_model()

if model is None:
    st.warning("⚠️ No trained model available. Please train a model first using the sidebar option.")
    st.info("📋 Make sure you have the 'enron_spam_data.csv' file in your directory before training.")
    st.stop()

# Main content
st.markdown('<div class="section-header">📧 Email Configuration</div>', unsafe_allow_html=True)

# Email configuration
col1, col2 = st.columns(2)

with col1:
    email_address = st.text_input("📧 Email Address", placeholder="your.email@gmail.com")
    password = st.text_input("🔐 Password", type="password", placeholder="App-specific password for Gmail")

    # Gmail App Password Instructions
    if "gmail.com" in str(email_address).lower():
        st.info("🔔 **Gmail Users**: Use App Password, not your regular password!")
        with st.expander("📋 How to generate Gmail App Password"):
            st.markdown("""
            **Follow these steps:**
            1. Go to [Google Account Settings](https://myaccount.google.com/)
            2. Click on **Security** in the left sidebar
            3. Under "How you sign in to Google", click **2-Step Verification** (enable if not already)
            4. Scroll down and click **App passwords**
            5. Select app: **Mail**
            6. Select device: **Other (Custom name)**
            7. Enter name: **Spam Detector**
            8. Click **Generate**
            9. Copy the 16-character password and use it here

            **Note**: Regular Gmail passwords won't work for IMAP access.
            """)

with col2:
    imap_server = st.selectbox("🌐 IMAP Server", [
        "imap.gmail.com",
        "imap.yahoo.com",
        "imap.outlook.com",
        "imap.aol.com",
        "Custom"
    ])

    if imap_server == "Custom":
        imap_server = st.text_input("Custom IMAP Server", placeholder="imap.your-provider.com")

    max_emails = st.slider("📊 Maximum Emails to Fetch", 10, 200, 50)

# Fetch emails button
if st.button("🔄 Fetch Emails", type="primary"):
    if not email_address or not password:
        st.error("Please provide email address and password")
    else:
        with st.spinner(f"Fetching emails from {email_address}..."):
            emails_df = fetch_emails(email_address, password, imap_server, max_emails=max_emails)

            if not emails_df.empty:
                st.session_state.emails_df = emails_df
                st.success(f"✅ Successfully fetched {len(emails_df)} emails!")
            else:
                st.error("❌ Failed to fetch emails or no emails found")

# Display emails if available
if not st.session_state.emails_df.empty:
    st.markdown('<div class="section-header">📋 Fetched Emails</div>', unsafe_allow_html=True)

    # Email analysis controls
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.write(f"Total emails: **{len(st.session_state.emails_df)}**")

    with col2:
        if st.button("✅ Select All"):
            st.session_state.emails_df['Selected'] = True

    with col3:
        if st.button("❌ Deselect All"):
            st.session_state.emails_df['Selected'] = False

    # Check spam button
    if st.button("🔍 Check Selected Emails for Spam", type="primary"):
        selected_emails = st.session_state.emails_df[st.session_state.emails_df['Selected']]

        if len(selected_emails) == 0:
            st.warning("Please select at least one email to check")
        else:
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, row in selected_emails.iterrows():
                spam_status = predict_spam(row['Full_Content'], model, vectorizer, scaler)
                st.session_state.emails_df.at[idx, 'Spam_Status'] = spam_status

                progress = (idx - selected_emails.index[0] + 1) / len(selected_emails)
                progress_bar.progress(progress)
                status_text.text(f'Checking email {idx - selected_emails.index[0] + 1}/{len(selected_emails)}')

            progress_bar.empty()
            status_text.empty()
            st.success(f"✅ Checked {len(selected_emails)} emails for spam!")

    # Display emails in a more user-friendly format
    st.markdown("### 📧 Email List")

    # View toggle
    view_mode = st.radio("Select View Mode", ["Table View", "Card View"], horizontal=True)

    if view_mode == "Table View":
        # Create a copy for display
        display_df = st.session_state.emails_df.copy()

        # Format the display
        display_df = display_df[['Selected', 'Subject', 'Sender', 'Receiver', 'Date', 'Content', 'Spam_Status']]

        # Use st.data_editor for interactive table
        edited_df = st.data_editor(
            display_df,
            use_container_width=True,
            num_rows="dynamic",
            column_config={
                "Selected": st.column_config.CheckboxColumn(
                    "Select",
                    help="Select emails to check for spam",
                    default=False,
                ),
                "Subject": st.column_config.TextColumn(
                    "Subject",
                    help="Email subject line",
                    max_chars=50,
                ),
                "Sender": st.column_config.TextColumn(
                    "From",
                    help="Sender email address",
                    max_chars=30,
                ),
                "Receiver": st.column_config.TextColumn(
                    "To",
                    help="Receiver email address",
                    max_chars=30,
                ),
                "Content": st.column_config.TextColumn(
                    "Preview",
                    help="Email content preview",
                    max_chars=100,
                ),
                "Spam_Status": st.column_config.TextColumn(
                    "Spam Status",
                    help="Spam detection result",
                )
            },
            hide_index=True,
        )

        # Update session state with edited selections
        st.session_state.emails_df['Selected'] = edited_df['Selected']

    else:  # Card View
        for idx, row in st.session_state.emails_df.iterrows():
            # Determine spam status styling
            if row['Spam_Status'] == 'Spam':
                status_class = 'spam'
            elif row['Spam_Status'] == 'Ham':
                status_class = 'ham'
            else:
                status_class = 'not-checked'

            # Create email card
            with st.container():
                col1, col2, col3 = st.columns([0.1, 0.8, 0.1])

                with col1:
                    selected = st.checkbox("", value=row['Selected'], key=f"select_{idx}")
                    st.session_state.emails_df.at[idx, 'Selected'] = selected

                with col2:
                    st.markdown(f"""
                    <div class="email-card">
                        <h4>📧 {row['Subject']}</h4>
                        <p><strong>From:</strong> {row['Sender']}</p>
                        <p><strong>To:</strong> {row['Receiver']}</p>
                        <p><strong>Date:</strong> {row['Date']}</p>
                        <p><strong>Content:</strong> {row['Content']}</p>
                    </div>
                    """, unsafe_allow_html=True)

                with col3:
                    st.markdown(f"""
                    <div class="spam-indicator {status_class}">
                        {row['Spam_Status']}
                    </div>
                    """, unsafe_allow_html=True)

    # Statistics section
    if st.session_state.emails_df['Spam_Status'].str.contains('Spam|Ham').any():
        st.markdown('<div class="section-header">📊 Statistics</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)

        spam_count = (st.session_state.emails_df['Spam_Status'] == 'Spam').sum()
        ham_count = (st.session_state.emails_df['Spam_Status'] == 'Ham').sum()
        not_checked = (st.session_state.emails_df['Spam_Status'] == 'Not Checked').sum()

        with col1:
            st.metric("Total Emails", len(st.session_state.emails_df))
        with col2:
            st.metric("Spam Emails", spam_count)
        with col3:
            st.metric("Ham Emails", ham_count)
        with col4:
            st.metric("Not Checked", not_checked)

# Manual email testing section
st.markdown('<div class="section-header">🧪 Manual Email Testing</div>', unsafe_allow_html=True)

manual_email = st.text_area("📝 Paste email content here to test:", height=150)

if st.button("🔍 Test This Email"):
    if manual_email.strip():
        result = predict_spam(manual_email, model, vectorizer, scaler)
        if result == "Spam":
            st.error(f"🚨 This email is classified as: **{result}**")
        else:
            st.success(f"✅ This email is classified as: **{result}**")
    else:
        st.warning("Please enter some email content to test.")

# Footer
st.markdown("---")
st.markdown("💡 **Important Setup Tips:**")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**📧 Gmail Setup:**")
    st.markdown("1. Enable 2-Factor Authentication")
    st.markdown("2. Go to Google Account → Security")
    st.markdown("3. Click 'App passwords'")
    st.markdown("4. Generate password for 'Mail'")
    st.markdown("5. Use 16-character app password here")

with col2:
    st.markdown("**🔧 Other Providers:**")
    st.markdown("• **Yahoo**: Enable 'Less secure app access'")
    st.markdown("• **Outlook**: Use regular password")
    st.markdown("• **Custom**: Check IMAP settings with provider")

st.markdown("**🚨 Common Issues:**")
st.markdown("• Gmail: Must use App Password, not regular password")
st.markdown("• Yahoo: Enable IMAP in settings")
st.markdown("• Outlook: Sometimes requires app password")
st.markdown("• Check firewall/antivirus blocking IMAP connections")



# With complete working with mail upload in chat

# import os
# import pandas as pd
# import numpy as np
# import joblib
# import streamlit as st
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import MinMaxScaler
# from stable_baselines3 import DQN
# from stable_baselines3.common.env_util import make_vec_env
# import gymnasium as gym
# from gymnasium import spaces
#
# # File paths
# MODEL_PATH = "spam_dqn_model"
# VECTORIZER_PATH = "tfidf_vectorizer.pkl"
# SCALER_PATH = "scaler.pkl"
# DATA_PATH = "enron_spam_data.csv"
#
# # Step 1: Load Dataset
# df = pd.read_csv(DATA_PATH)
# df.dropna(subset=['Message', 'Spam/Ham'], inplace=True)
# df['label'] = df['Spam/Ham'].map({'ham': 0, 'spam': 1})
# X_text = df['Message']
# y = df['label'].values
#
# # Step 2: TF-IDF and Scaling
# if os.path.exists(VECTORIZER_PATH) and os.path.exists(SCALER_PATH):
#     vectorizer = joblib.load(VECTORIZER_PATH)
#     scaler = joblib.load(SCALER_PATH)
#     X = vectorizer.transform(X_text).toarray()
#     X = scaler.transform(X)
# else:
#     vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
#     X = vectorizer.fit_transform(X_text).toarray()
#     scaler = MinMaxScaler()
#     X = scaler.fit_transform(X)
#     joblib.dump(vectorizer, VECTORIZER_PATH)
#     joblib.dump(scaler, SCALER_PATH)
#
# # Step 3: Train/Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Step 4: Define Custom Gym Environment
# class SpamEnv(gym.Env):
#     def __init__(self, X, y):
#         super(SpamEnv, self).__init__()
#         self.X = X.astype(np.float32)
#         self.y = y
#         self.current_index = 0
#         self.max_steps = len(X)
#         self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(X.shape[1],), dtype=np.float32)
#         self.action_space = spaces.Discrete(2)
#
#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.current_index = 0
#         obs = self.X[self.current_index]
#         return obs, {}
#
#     def step(self, action):
#         true_label = self.y[self.current_index]
#         reward = 1.0 if action == true_label else -1.0
#         self.current_index += 1
#         terminated = self.current_index >= self.max_steps
#         next_obs = self.X[self.current_index] if not terminated else np.zeros(self.X.shape[1], dtype=np.float32)
#         return next_obs, reward, terminated, False, {}
#
# # Step 5: Train or Load Model
# if os.path.exists(MODEL_PATH + ".zip"):
#     model = DQN.load(MODEL_PATH)
# else:
#     def make_env():
#         return SpamEnv(X_train, y_train)
#
#     env = make_vec_env(make_env, n_envs=1)
#
#     model = DQN(
#         "MlpPolicy",
#         env,
#         verbose=1,
#         learning_rate=0.0001,
#         buffer_size=10000,
#         learning_starts=1000,
#         batch_size=32,
#         gamma=0.99,
#         exploration_fraction=0.1,
#         exploration_initial_eps=1.0,
#         exploration_final_eps=0.02,
#         train_freq=4,
#         target_update_interval=1000
#     )
#
#     model.learn(total_timesteps=50000)
#     model.save(MODEL_PATH)
#
# # Step 6: Prediction Function
# def predict_email_spam(email_text):
#     email_vector = vectorizer.transform([email_text]).toarray()
#     email_vector = scaler.transform(email_vector)
#     action, _ = model.predict(email_vector, deterministic=True)
#     return "Spam" if action == 1 else "Ham"
#
# # Step 7: Streamlit UI
# st.set_page_config(page_title="Spam Email Detector", layout="centered")
# st.title("📧 Spam Email Detector")
# st.write("Paste your email content below to check if it's spam or not:")
#
# email_input = st.text_area("Email Content", height=200)
#
# if st.button("Check Spam"):
#     if email_input.strip() == "":
#         st.warning("Please enter some email content.")
#     else:
#         result = predict_email_spam(email_input)
#         st.success(f"The email is classified as: **{result}**")


# without ui

# import os
# import pandas as pd
# import numpy as np
# import joblib
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import MinMaxScaler
# from stable_baselines3 import DQN
# from stable_baselines3.common.env_util import make_vec_env
# import gymnasium as gym
# from gymnasium import spaces
#
# # File paths
# MODEL_PATH = "spam_dqn_model"
# VECTORIZER_PATH = "tfidf_vectorizer.pkl"
# SCALER_PATH = "scaler.pkl"
# DATA_PATH = "enron_spam_data.csv"
#
# # Step 1: Load Dataset
# df = pd.read_csv(DATA_PATH)
# df.dropna(subset=['Message', 'Spam/Ham'], inplace=True)
# df['label'] = df['Spam/Ham'].map({'ham': 0, 'spam': 1})
# X_text = df['Message']
# y = df['label'].values
#
# # Step 2: TF-IDF and Scaling
# if os.path.exists(VECTORIZER_PATH) and os.path.exists(SCALER_PATH):
#     vectorizer = joblib.load(VECTORIZER_PATH)
#     scaler = joblib.load(SCALER_PATH)
#     X = vectorizer.transform(X_text).toarray()
#     X = scaler.transform(X)
# else:
#     vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
#     X = vectorizer.fit_transform(X_text).toarray()
#     scaler = MinMaxScaler()
#     X = scaler.fit_transform(X)
#     joblib.dump(vectorizer, VECTORIZER_PATH)
#     joblib.dump(scaler, SCALER_PATH)
#
# # Step 3: Train/Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Step 4: Define Custom Gym Environment
# class SpamEnv(gym.Env):
#     def __init__(self, X, y):
#         super(SpamEnv, self).__init__()
#         self.X = X.astype(np.float32)
#         self.y = y
#         self.current_index = 0
#         self.max_steps = len(X)
#         self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(X.shape[1],), dtype=np.float32)
#         self.action_space = spaces.Discrete(2)
#
#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.current_index = 0
#         obs = self.X[self.current_index]
#         return obs, {}
#
#     def step(self, action):
#         true_label = self.y[self.current_index]
#         reward = 1.0 if action == true_label else -1.0
#         self.current_index += 1
#         terminated = self.current_index >= self.max_steps
#         next_obs = self.X[self.current_index] if not terminated else np.zeros(self.X.shape[1], dtype=np.float32)
#         return next_obs, reward, terminated, False, {}
#
# # Step 5: Train or Load Model
# if os.path.exists(MODEL_PATH + ".zip"):
#     model = DQN.load(MODEL_PATH)
# else:
#     def make_env():
#         return SpamEnv(X_train, y_train)
#
#     env = make_vec_env(make_env, n_envs=1)
#
#     model = DQN(
#         "MlpPolicy",
#         env,
#         verbose=1,
#         learning_rate=0.0001,
#         buffer_size=10000,
#         learning_starts=1000,
#         batch_size=32,
#         gamma=0.99,
#         exploration_fraction=0.1,
#         exploration_initial_eps=1.0,
#         exploration_final_eps=0.02,
#         train_freq=4,
#         target_update_interval=1000
#     )
#
#     model.learn(total_timesteps=50000)
#     model.save(MODEL_PATH)
#
# # Step 6: Predict New Email
# def predict_email_spam(email_text):
#     email_vector = vectorizer.transform([email_text]).toarray()
#     email_vector = scaler.transform(email_vector)
#     action, _ = model.predict(email_vector, deterministic=True)
#     return "Spam" if action == 1 else "Ham"
#
# # Step 7: User Input
# user_email = input("Enter your email content:\n")
# result = predict_email_spam(user_email)
# print(f"\nThe email is classified as: {result}")


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import MinMaxScaler
# from stable_baselines3 import DQN
# from stable_baselines3.common.env_util import make_vec_env
# import gymnasium as gym
# from gymnasium import spaces
#
# # Step 1: Load Enron Dataset
# df = pd.read_csv("enron_spam_data.csv")
#
# # Step 2: Preprocess Emails
# df.dropna(subset=['Message', 'Spam/Ham'], inplace=True)
# df['label'] = df['Spam/Ham'].map({'ham': 0, 'spam': 1})
# X_text = df['Message']
# y = df['label'].values
#
# # Step 3: TF-IDF Vectorization
# vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
# X = vectorizer.fit_transform(X_text).toarray()
#
# # Step 4: Normalize Features
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
#
# # Step 5: Train/Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Step 6: Define Custom Gym Environment
# class SpamEnv(gym.Env):
#     def __init__(self, X, y):
#         super(SpamEnv, self).__init__()
#         self.X = X.astype(np.float32)
#         self.y = y
#         self.current_index = 0
#         self.max_steps = len(X)
#         self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(X.shape[1],), dtype=np.float32)
#         self.action_space = spaces.Discrete(2)
#
#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.current_index = 0
#         obs = self.X[self.current_index]
#         return obs, {}
#
#     def step(self, action):
#         true_label = self.y[self.current_index]
#         reward = 1.0 if action == true_label else -1.0
#         self.current_index += 1
#         terminated = self.current_index >= self.max_steps
#         next_obs = self.X[self.current_index] if not terminated else np.zeros(self.X.shape[1], dtype=np.float32)
#         return next_obs, reward, terminated, False, {}
#
# # Step 7: Train DQN Agent
# def make_env():
#     return SpamEnv(X_train, y_train)
#
# env = make_vec_env(make_env, n_envs=1)
#
# model = DQN(
#     "MlpPolicy",
#     env,
#     verbose=1,
#     learning_rate=0.0001,
#     buffer_size=10000,
#     learning_starts=1000,
#     batch_size=32,
#     gamma=0.99,
#     exploration_fraction=0.1,
#     exploration_initial_eps=1.0,
#     exploration_final_eps=0.02,
#     train_freq=4,
#     target_update_interval=1000
# )
#
# model.learn(total_timesteps=50000)
#
# # Step 8: Predict New Email
# def predict_email_spam(email_text):
#     email_vector = vectorizer.transform([email_text]).toarray()
#     email_vector = scaler.transform(email_vector)
#     action, _ = model.predict(email_vector, deterministic=True)
#     return "Spam" if action == 1 else "Ham"
#
# # Step 9: User Input
# user_email = input("Enter your email content:\n")
# result = predict_email_spam(user_email)
# print(f"\nThe email is classified as: {result}")


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import MinMaxScaler
# from stable_baselines3 import DQN
# from stable_baselines3.common.env_util import make_vec_env
# import gymnasium as gym
# from gymnasium import spaces
# from sklearn.metrics import classification_report, confusion_matrix
#
# # Step 1: Load Enron Dataset
# try:
#     df = pd.read_csv("enron_spam_data.csv")
#     print(f"Dataset loaded successfully. Shape: {df.shape}")
# except FileNotFoundError:
#     print("Error: enron_spam_data.csv not found. Please check the file path.")
#     exit()
#
# # Step 2: Preprocess Emails
# df.dropna(subset=['Message', 'Spam/Ham'], inplace=True)
# df['label'] = df['Spam/Ham'].map({'ham': 0, 'spam': 1})
# X_text = df['Message']
# y = df['label'].values
#
# # Step 3: TF-IDF Vectorization
# vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
# X = vectorizer.fit_transform(X_text).toarray()
#
# # Step 4: Normalize Features
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
#
# # Step 5: Train/Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Step 6: Define Custom Gym Environment
# class SpamEnv(gym.Env):
#     def __init__(self, X, y):
#         super(SpamEnv, self).__init__()
#         self.X = X.astype(np.float32)
#         self.y = y
#         self.current_index = 0
#         self.max_steps = len(X)
#         self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(X.shape[1],), dtype=np.float32)
#         self.action_space = spaces.Discrete(2)
#
#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.current_index = 0
#         obs = self.X[self.current_index]
#         return obs, {}
#
#     def step(self, action):
#         true_label = self.y[self.current_index]
#         reward = 1.0 if action == true_label else -1.0
#         self.current_index += 1
#         terminated = self.current_index >= self.max_steps
#         next_obs = self.X[self.current_index] if not terminated else np.zeros(self.X.shape[1], dtype=np.float32)
#         return next_obs, reward, terminated, False, {}
#
# # Step 7: Train DQN Agent
# def make_env():
#     return SpamEnv(X_train, y_train)
#
# env = make_vec_env(make_env, n_envs=1)
#
# model = DQN(
#     "MlpPolicy",
#     env,
#     verbose=1,
#     learning_rate=0.0001,
#     buffer_size=10000,
#     learning_starts=1000,
#     batch_size=32,
#     gamma=0.99,
#     exploration_fraction=0.1,
#     exploration_initial_eps=1.0,
#     exploration_final_eps=0.02,
#     train_freq=4,
#     target_update_interval=1000
# )
#
# model.learn(total_timesteps=50000, progress_bar=True)
#
# # Step 8: Evaluate the Model
# test_env = SpamEnv(X_test, y_test)
# obs, _ = test_env.reset()
#
# correct_predictions = 0
# total_predictions = 0
# predictions = []
# true_labels = []
#
# for i in range(len(X_test)):
#     action, _ = model.predict(obs, deterministic=True)
#     action = int(action)
#     true_label = y_test[i]
#     predictions.append(action)
#     true_labels.append(true_label)
#     if action == true_label:
#         correct_predictions += 1
#     total_predictions += 1
#     obs, _, terminated, _, _ = test_env.step(action)
#     if terminated:
#         break
#
# accuracy = correct_predictions / total_predictions
# print(f"\nTest Accuracy: {accuracy:.4f}")
# print(classification_report(true_labels, predictions, target_names=['Ham', 'Spam']))
# print(confusion_matrix(true_labels, predictions))
#
# # Step 9: Predict New Email
# def predict_email_spam(email_text, vectorizer, scaler, model):
#     email_vector = vectorizer.transform([email_text]).toarray()
#     email_vector = scaler.transform(email_vector)
#     action, _ = model.predict(email_vector, deterministic=True)
#     return "Spam" if action == 1 else "Ham"
#
# # Example usage
# new_email = "Congratulations! You've won a $1000 gift card. Click here to claim your prize."
# result = predict_email_spam(new_email, vectorizer, scaler, model)
# print(f"\nNew email prediction: {result}")


# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.preprocessing import MinMaxScaler
# from stable_baselines3 import DQN
# from stable_baselines3.common.env_util import make_vec_env
# import gymnasium as gym
# from gymnasium import spaces
#
# # Step 1: Load Enron Dataset
# # Note: Make sure the CSV file path is correct
# try:
#     df = pd.read_csv("enron_spam_data.csv")
#     print(f"Dataset loaded successfully. Shape: {df.shape}")
# except FileNotFoundError:
#     print("Error: enron_spam_data.csv not found. Please check the file path.")
#     exit()
#
# # Step 2: Preprocess Emails
# df.dropna(subset=['Message', 'Spam/Ham'], inplace=True)
# df['label'] = df['Spam/Ham'].map({'ham': 0, 'spam': 1})
# X_text = df['Message']
# y = df['label'].values
#
# print(f"Data after preprocessing: {len(df)} emails")
# print(f"Ham: {sum(y == 0)}, Spam: {sum(y == 1)}")
#
# # Step 3: TF-IDF Vectorization
# vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
# X = vectorizer.fit_transform(X_text).toarray()
#
# # Step 4: Normalize Features
# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)
#
# # Step 5: Train/Test Split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# print(f"Training set: {X_train.shape[0]} samples")
# print(f"Test set: {X_test.shape[0]} samples")
#
#
# # Step 6: Define Custom Gym Environment
# class SpamEnv(gym.Env):
#     def __init__(self, X, y):
#         super(SpamEnv, self).__init__()
#         self.X = X.astype(np.float32)
#         self.y = y
#         self.current_index = 0
#         self.max_steps = len(X)
#
#         # Define observation and action spaces
#         self.observation_space = spaces.Box(
#             low=0.0, high=1.0, shape=(X.shape[1],), dtype=np.float32
#         )
#         self.action_space = spaces.Discrete(2)  # 0 = ham, 1 = spam
#
#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         self.current_index = 0
#         obs = self.X[self.current_index]
#         return obs, {}
#
#     def step(self, action):
#         # Get current label
#         true_label = self.y[self.current_index]
#
#         # Calculate reward
#         if action == true_label:
#             reward = 1.0  # Correct classification
#         else:
#             reward = -1.0  # Incorrect classification
#
#         # Move to next sample
#         self.current_index += 1
#         terminated = self.current_index >= self.max_steps
#
#         # Get next observation
#         if not terminated:
#             next_obs = self.X[self.current_index]
#         else:
#             next_obs = np.zeros(self.X.shape[1], dtype=np.float32)
#
#         return next_obs, reward, terminated, False, {}
#
#
# # Step 7: Train DQN Agent
# print("Creating environment and training DQN agent...")
#
#
# # Create vectorized environment for better performance
# def make_env():
#     return SpamEnv(X_train, y_train)
#
#
# env = make_vec_env(make_env, n_envs=1)
#
# # Create DQN model with appropriate hyperparameters
# model = DQN(
#     "MlpPolicy",
#     env,
#     verbose=1,
#     learning_rate=0.0001,
#     buffer_size=10000,
#     learning_starts=1000,
#     batch_size=32,
#     gamma=0.99,
#     exploration_fraction=0.1,
#     exploration_initial_eps=1.0,
#     exploration_final_eps=0.02,
#     train_freq=4,
#     target_update_interval=1000
# )
#
# # Train the model
# print("Starting training...")
# model.learn(total_timesteps=50000, progress_bar=True)
#
# # Step 8: Evaluate the Model
# print("Evaluating the model...")
#
# # Create test environment
# test_env = SpamEnv(X_test, y_test)
# obs, _ = test_env.reset()
#
# correct_predictions = 0
# total_predictions = 0
# predictions = []
# true_labels = []
#
# for i in range(len(X_test)):
#     # Get action from trained model
#     action, _ = model.predict(obs, deterministic=True)
#     action = int(action)
#
#     # Get true label
#     true_label = y_test[i]
#
#     # Store predictions
#     predictions.append(action)
#     true_labels.append(true_label)
#
#     # Check if prediction is correct
#     if action == true_label:
#         correct_predictions += 1
#
#     total_predictions += 1
#
#     # Take step in environment
#     obs, reward, terminated, truncated, _ = test_env.step(action)
#
#     if terminated:
#         break
#
# # Calculate metrics
# accuracy = correct_predictions / total_predictions
# print(f"\nTest Results:")
# print(f"Accuracy: {accuracy:.4f}")
# print(f"Correct predictions: {correct_predictions}/{total_predictions}")
#
# # Additional metrics
# from sklearn.metrics import classification_report, confusion_matrix
#
# print(f"\nDetailed Classification Report:")
# print(classification_report(true_labels, predictions, target_names=['Ham', 'Spam']))
#
# print(f"\nConfusion Matrix:")
# print(confusion_matrix(true_labels, predictions))
#
# # Compare with simple baseline (always predict majority class)
# majority_class = np.bincount(y_test).argmax()
# baseline_accuracy = np.mean(y_test == majority_class)
# print(f"\nBaseline accuracy (majority class): {baseline_accuracy:.4f}")
#
# print(f"DQN improvement over baseline: {accuracy - baseline_accuracy:.4f}")