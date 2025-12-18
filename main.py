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
import json
import html

# File paths
MODEL_DIR = "models"

# Create models directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

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
if 'current_model_name' not in st.session_state:
    st.session_state.current_model_name = None


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


def process_uploaded_dataset(uploaded_file):
    """Process uploaded dataset file"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension == 'csv':
            df = pd.read_csv(uploaded_file)
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV, Excel, or JSON files.")
            return None

        st.write("Dataset Preview:")
        st.dataframe(df.head())
        st.write(f"Dataset shape: {df.shape}")
        st.write(f"Columns: {list(df.columns)}")

        return df
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None


def train_model_from_dataset(df, text_column, label_column, model_name, timesteps=50000):
    """Train the model with user-provided dataset"""
    try:
        # Debug prints to check types
        st.write(f"Type of text_column: {type(text_column)}, Value: {text_column}")
        st.write(f"Type of label_column: {type(label_column)}, Value: {label_column}")

        # Ensure columns are single strings
        if isinstance(text_column, list):
            if len(text_column) != 1:
                st.error("Please select only one text column.")
                return False
            text_column = text_column[0]

        if isinstance(label_column, list):
            if len(label_column) != 1:
                st.error("Please select only one label column.")
                return False
            label_column = label_column[0]

        # Validate columns
        if text_column not in df.columns or label_column not in df.columns:
            st.error(f"Selected columns not found in dataset")
            return False

        # Prepare data
        df_clean = df[[text_column, label_column]].dropna()

        # Map labels to 0 and 1
        unique_labels = df_clean[label_column].unique()
        st.write(f"Unique labels found: {unique_labels}")

        if len(unique_labels) != 2:
            st.error("Dataset must have exactly 2 classes (spam/ham)")
            return False

        # Create label mapping
        label_map = {unique_labels[0]: 0, unique_labels[1]: 1}
        df_clean['label'] = df_clean[label_column].map(label_map)

        X_text = df_clean[text_column]
        y = df_clean['label'].values

        st.info(f"Training data: {len(df_clean)} samples")
        st.info(f"Class 0: {sum(y == 0)}, Class 1: {sum(y == 1)}")

        # TF-IDF and Scaling
        vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
        X = vectorizer.fit_transform(X_text).toarray()
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

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

        # Train the model with progress
        progress_bar = st.progress(0)
        status_text = st.empty()

        steps_per_update = timesteps // 10
        for i in range(10):
            model.learn(total_timesteps=steps_per_update)
            progress_bar.progress((i + 1) / 10)
            status_text.text(f'Training progress: {((i + 1) / 10) * 100:.0f}%')

        # Evaluate on test set
        test_env = SpamEnv(X_test, y_test)
        obs, _ = test_env.reset()
        correct = 0
        total = len(X_test)

        for i in range(total):
            action, _ = model.predict(obs, deterministic=True)
            if int(action) == y_test[i]:
                correct += 1
            obs, _, terminated, _, _ = test_env.step(int(action))
            if terminated:
                break

        accuracy = correct / total
        st.success(f"Model trained! Test Accuracy: {accuracy:.2%}")

        # Save model, vectorizer, and scaler
        model_path = os.path.join(MODEL_DIR, f"{model_name}")
        vectorizer_path = os.path.join(MODEL_DIR, f"{model_name}_vectorizer.pkl")
        scaler_path = os.path.join(MODEL_DIR, f"{model_name}_scaler.pkl")
        metadata_path = os.path.join(MODEL_DIR, f"{model_name}_metadata.json")

        model.save(model_path)
        joblib.dump(vectorizer, vectorizer_path)
        joblib.dump(scaler, scaler_path)

        # Save metadata
        metadata = {
            'model_name': model_name,
            'accuracy': accuracy,
            'training_samples': len(df_clean),
            'timestamp': datetime.datetime.now().isoformat(),
            'label_map': {str(k): v for k, v in label_map.items()}
        }
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Update session state
        st.session_state.model = model
        st.session_state.vectorizer = vectorizer
        st.session_state.scaler = scaler
        st.session_state.model_loaded = True
        st.session_state.current_model_name = model_name

        progress_bar.empty()
        status_text.empty()

        return True

    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return False


def load_saved_model(model_name):
    """Load a saved model"""
    try:
        model_path = os.path.join(MODEL_DIR, f"{model_name}.zip")
        vectorizer_path = os.path.join(MODEL_DIR, f"{model_name}_vectorizer.pkl")
        scaler_path = os.path.join(MODEL_DIR, f"{model_name}_scaler.pkl")
        metadata_path = os.path.join(MODEL_DIR, f"{model_name}_metadata.json")

        if not all(os.path.exists(p) for p in [model_path, vectorizer_path, scaler_path]):
            st.error("Model files not found")
            return False

        model = DQN.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        scaler = joblib.load(scaler_path)

        # Load metadata if exists
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            st.info(
                f"Model Info: Accuracy: {metadata.get('accuracy', 'N/A'):.2%}, Training samples: {metadata.get('training_samples', 'N/A')}")

        st.session_state.model = model
        st.session_state.vectorizer = vectorizer
        st.session_state.scaler = scaler
        st.session_state.model_loaded = True
        st.session_state.current_model_name = model_name

        return True
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return False


def get_saved_models():
    """Get list of saved models"""
    if not os.path.exists(MODEL_DIR):
        return []

    models = []
    for file in os.listdir(MODEL_DIR):
        if file.endswith('.zip'):
            model_name = file.replace('.zip', '')
            models.append(model_name)

    return models


def clean_text(text):
    """Clean email text for better processing"""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'<[^>]+>', '', text)
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
        context = ssl.create_default_context()
        socket.setdefaulttimeout(30)
        mail = imaplib.IMAP4_SSL(imap_server, port, ssl_context=context)

        try:
            mail.login(email_address, password)
        except imaplib.IMAP4.error as login_error:
            error_msg = str(login_error)
            if "Application-specific password required" in error_msg or "ALERT" in error_msg:
                raise Exception("Gmail requires App-Specific Password")
            elif "authentication failed" in error_msg.lower():
                raise Exception("Authentication failed. Check your credentials")
            else:
                raise Exception(f"Login failed: {error_msg}")

        status, select_result = mail.select('inbox')
        if status != 'OK':
            raise Exception(f"Failed to select inbox")

        status, messages = mail.search(None, 'ALL')
        if status != 'OK':
            raise Exception("Failed to search emails")

        message_ids = messages[0].split()
        if not message_ids:
            raise Exception("No emails found")

        message_ids = message_ids[-max_emails:]
        emails_data = []

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, msg_id in enumerate(message_ids):
            try:
                status, msg_data = mail.fetch(msg_id, '(RFC822)')
                if status != 'OK':
                    continue

                email_body = msg_data[0][1]
                email_message = email.message_from_bytes(email_body)

                subject = decode_mime_words(email_message["Subject"]) or "No Subject"
                sender = decode_mime_words(email_message["From"]) or "Unknown"
                receiver = decode_mime_words(email_message["To"]) or email_address
                date_str = email_message["Date"] or ""

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

                progress = (i + 1) / len(message_ids)
                progress_bar.progress(progress)
                status_text.text(f'Fetching: {i + 1}/{len(message_ids)}')

            except:
                continue

        mail.close()
        mail.logout()

        progress_bar.empty()
        status_text.empty()

        if not emails_data:
            raise Exception("No emails could be processed")

        return pd.DataFrame(emails_data)

    except Exception as e:
        st.error(f"Error fetching emails: {str(e)}")
        return pd.DataFrame()


def predict_spam(content, model, vectorizer, scaler):
    """Predict if email content is spam"""
    try:
        if not content or pd.isna(content):
            return "Unknown"

        content_vector = vectorizer.transform([str(content)]).toarray()
        content_vector = scaler.transform(content_vector)
        action, _ = model.predict(content_vector, deterministic=True)
        return "Spam" if action == 1 else "Ham"
    except:
        return "Error"


# Streamlit UI
st.set_page_config(page_title="Advanced Spam Detector", layout="wide")

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

st.markdown('<div class="main-header">Advanced Spam Email Detector</div>', unsafe_allow_html=True)

# Sidebar for model management
st.sidebar.markdown("### Model Management")

# Tab selection
model_tab = st.sidebar.radio("Select Action", ["Train New Model", "Load Saved Model", "Fine-tune Model"])

if model_tab == "Train New Model":
    st.sidebar.markdown("#### Upload Dataset")
    uploaded_file = st.sidebar.file_uploader("Upload CSV/Excel/JSON", type=['csv', 'xlsx', 'xls', 'json'])

    if uploaded_file:
        df = process_uploaded_dataset(uploaded_file)

        if df is not None:
            text_col = st.sidebar.selectbox("Select Text Column", df.columns)
            label_col = st.sidebar.selectbox("Select Label Column", df.columns)
            model_name = st.sidebar.text_input("Model Name", "spam_model")
            timesteps = st.sidebar.slider("Training Steps", 10000, 100000, 50000, 10000)

            if st.sidebar.button("Train Model"):
                with st.spinner("Training model..."):
                    train_model_from_dataset(df, text_col, label_col, model_name, timesteps)

elif model_tab == "Load Saved Model":
    saved_models = get_saved_models()

    if saved_models:
        selected_model = st.sidebar.selectbox("Select Model", saved_models)
        if st.sidebar.button("Load Model"):
            with st.spinner("Loading model..."):
                load_saved_model(selected_model)
    else:
        st.sidebar.info("No saved models found. Train a model first.")

elif model_tab == "Fine-tune Model":
    if st.session_state.model_loaded:
        st.sidebar.info(f"Current model: {st.session_state.current_model_name}")
        uploaded_file = st.sidebar.file_uploader("Upload Additional Data", type=['csv', 'xlsx', 'xls', 'json'])

        if uploaded_file:
            df = process_uploaded_dataset(uploaded_file)
            if df is not None:
                text_col = st.sidebar.selectbox("Text Column", df.columns)
                label_col = st.sidebar.selectbox("Label Column", df.columns)
                timesteps = st.sidebar.slider("Fine-tune Steps", 5000, 50000, 10000, 5000)

                if st.sidebar.button("Fine-tune"):
                    new_model_name = f"{st.session_state.current_model_name}_finetuned"
                    with st.spinner("Fine-tuning..."):
                        train_model_from_dataset(df, text_col, label_col, new_model_name, timesteps)
    else:
        st.sidebar.warning("Load a model first")

# Show current model status
if st.session_state.model_loaded:
    st.sidebar.success(f"Model loaded: {st.session_state.current_model_name}")
else:
    st.sidebar.warning("No model loaded")

# Main content - Email fetching
st.markdown('<div class="section-header">Email Configuration</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    email_address = st.text_input("Email Address", placeholder="your.email@gmail.com")
    password = st.text_input("Password", type="password", placeholder="App password")

with col2:
    imap_server = st.selectbox("IMAP Server", [
        "imap.gmail.com",
        "imap.yahoo.com",
        "imap.outlook.com",
        "Custom"
    ])

    if imap_server == "Custom":
        imap_server = st.text_input("Custom IMAP", placeholder="imap.provider.com")

    max_emails = st.slider("Max Emails", 10, 200, 50)

if st.button("Fetch Emails", type="primary"):
    if not email_address or not password:
        st.error("Provide email and password")
    elif not st.session_state.model_loaded:
        st.error("Load or train a model first")
    else:
        with st.spinner("Fetching..."):
            emails_df = fetch_emails(email_address, password, imap_server, max_emails=max_emails)
            if not emails_df.empty:
                st.session_state.emails_df = emails_df
                st.success(f"Fetched {len(emails_df)} emails!")

# Display emails
if not st.session_state.emails_df.empty and st.session_state.model_loaded:
    st.markdown('<div class="section-header">Fetched Emails</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.write(f"Total: **{len(st.session_state.emails_df)}**")
    with col2:
        if st.button("Select All"):
            st.session_state.emails_df['Selected'] = True
            st.rerun()
    with col3:
        if st.button("Deselect All"):
            st.session_state.emails_df['Selected'] = False
            st.rerun()

    if st.button("Check Selected for Spam", type="primary"):
        selected = st.session_state.emails_df[st.session_state.emails_df['Selected']]
        if len(selected) == 0:
            st.warning("Select at least one email")
        else:
            progress_bar = st.progress(0)
            for idx, (i, row) in enumerate(selected.iterrows()):
                spam_status = predict_spam(row['Full_Content'], st.session_state.model,
                                           st.session_state.vectorizer, st.session_state.scaler)
                st.session_state.emails_df.at[i, 'Spam_Status'] = spam_status
                progress_bar.progress((idx + 1) / len(selected))
            progress_bar.empty()
            st.success(f"Checked {len(selected)} emails!")
            st.rerun()

    view_mode = st.radio("View Mode", ["Table View", "Card View"], horizontal=True)

    if view_mode == "Table View":
        display_df = st.session_state.emails_df[
            ['Selected', 'Subject', 'Sender', 'Date', 'Content', 'Spam_Status']].copy()
        edited_df = st.data_editor(
            display_df,
            use_container_width=True,
            column_config={
                "Selected": st.column_config.CheckboxColumn("Select", default=False),
                "Spam_Status": st.column_config.TextColumn("Status")
            },
            hide_index=True,
        )
        st.session_state.emails_df['Selected'] = edited_df['Selected'].values

    else:  # Card View
        for idx, row in st.session_state.emails_df.iterrows():
            status_class = 'spam' if row['Spam_Status'] == 'Spam' else (
                'ham' if row['Spam_Status'] == 'Ham' else 'not-checked')

            col1, col2, col3 = st.columns([0.5, 8, 1.5])

            with col1:
                if st.checkbox("", value=row['Selected'], key=f"sel_{idx}"):
                    st.session_state.emails_df.at[idx, 'Selected'] = True
                else:
                    st.session_state.emails_df.at[idx, 'Selected'] = False

            with col2:
                st.markdown(f"""
                <div class="email-card">
                    <h4>{html.escape(row['Subject'])}</h4>
                    <p><strong>From:</strong> {html.escape(row['Sender'])}</p>
                    <p><strong>Date:</strong> {html.escape(row['Date'])}</p>
                    <p><strong>Preview:</strong> {html.escape(row['Content'])}</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f'<div class="spam-indicator {status_class}">{html.escape(row["Spam_Status"])}</div>',
                            unsafe_allow_html=True)

    # Statistics
    if st.session_state.emails_df['Spam_Status'].str.contains('Spam|Ham').any():
        st.markdown('<div class="section-header">Statistics</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)

        spam_count = (st.session_state.emails_df['Spam_Status'] == 'Spam').sum()
        ham_count = (st.session_state.emails_df['Spam_Status'] == 'Ham').sum()
        not_checked = (st.session_state.emails_df['Spam_Status'] == 'Not Checked').sum()

        col1.metric("Total", len(st.session_state.emails_df))
        col2.metric("Spam", spam_count)
        col3.metric("Ham", ham_count)
        col4.metric("Unchecked", not_checked)

# Manual testing
st.markdown('<div class="section-header">Manual Testing</div>', unsafe_allow_html=True)
manual_email = st.text_area("Test email content:", height=150)

if st.button("Test Email"):
    if manual_email.strip() and st.session_state.model_loaded:
        result = predict_spam(manual_email, st.session_state.model,
                              st.session_state.vectorizer, st.session_state.scaler)
        if result == "Spam":
            st.error(f"Classified as: **{result}**")
        else:
            st.success(f"Classified as: **{result}**")
    elif not st.session_state.model_loaded:
        st.warning("Load a model first")
    else:
        st.warning("Enter content to test")
