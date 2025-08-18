import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
from gymnasium import spaces

# File paths
MODEL_PATH = "spam_dqn_model"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"
SCALER_PATH = "scaler.pkl"
DATA_PATH = "enron_spam_data.csv"

# Step 1: Load Dataset
df = pd.read_csv(DATA_PATH)
df.dropna(subset=['Message', 'Spam/Ham'], inplace=True)
df['label'] = df['Spam/Ham'].map({'ham': 0, 'spam': 1})
X_text = df['Message']
y = df['label'].values

# Step 2: TF-IDF and Scaling
if os.path.exists(VECTORIZER_PATH) and os.path.exists(SCALER_PATH):
    vectorizer = joblib.load(VECTORIZER_PATH)
    scaler = joblib.load(SCALER_PATH)
    X = vectorizer.transform(X_text).toarray()
    X = scaler.transform(X)
else:
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    X = vectorizer.fit_transform(X_text).toarray()
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(scaler, SCALER_PATH)

# Step 3: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Define Custom Gym Environment
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

# Step 5: Train or Load Model
if os.path.exists(MODEL_PATH + ".zip"):
    model = DQN.load(MODEL_PATH)
else:
    def make_env():
        return SpamEnv(X_train, y_train)

    env = make_vec_env(make_env, n_envs=1)

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

    model.learn(total_timesteps=50000)
    model.save(MODEL_PATH)

# Step 6: Prediction Function
def predict_email_spam(email_text):
    email_vector = vectorizer.transform([email_text]).toarray()
    email_vector = scaler.transform(email_vector)
    action, _ = model.predict(email_vector, deterministic=True)
    return "Spam" if action == 1 else "Ham"

# Step 7: Streamlit UI
st.set_page_config(page_title="Spam Email Detector", layout="centered")
st.title("📧 Spam Email Detector")
st.write("Paste your email content below to check if it's spam or not:")

email_input = st.text_area("Email Content", height=200)

if st.button("Check Spam"):
    if email_input.strip() == "":
        st.warning("Please enter some email content.")
    else:
        result = predict_email_spam(email_input)
        st.success(f"The email is classified as: **{result}**")


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