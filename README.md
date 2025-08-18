# spam_Detector

Spam Email Detector using Reinforcement Learning
This project implements a spam email detector using a Deep Q-Network (DQN), a reinforcement learning algorithm. The model is trained on the Enron Spam Dataset and deployed as an interactive web application using Streamlit.

📖 How It Works
The project treats spam classification as a reinforcement learning problem. An "agent" is trained to decide whether an email is "Spam" or "Ham" (not spam).

Data Preprocessing: Email text is converted into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency). These vectors are then scaled to a range of [0, 1].

Custom Gym Environment: A custom environment (SpamEnv) compatible with the Gymnasium library is created.

State: The TF-IDF vector of an email.

Action: Classify the email as Ham (0) or Spam (1).

Reward: The agent receives a reward of +1 for a correct classification and -1 for an incorrect one.

DQN Model: A Deep Q-Network (DQN) model from the stable-baselines3 library is used. The model learns a policy to maximize the cumulative reward by correctly classifying emails.

Training: The model is trained on the preprocessed training data. The trained model, TF-IDF vectorizer, and scaler are saved to disk to be reused for inference.

Web Application: A simple UI is built with Streamlit that allows users to input email text and get a real-time prediction from the trained model.

🛠️ Technology Stack
Python: Core programming language.

Pandas & NumPy: For data manipulation and numerical operations.

Scikit-learn: For TF-IDF vectorization and data scaling.

Gymnasium: For creating the custom reinforcement learning environment.

Stable Baselines3: For implementing the DQN algorithm.

Streamlit: To create and serve the interactive web application.

Joblib: For saving and loading the trained vectorizer and scaler.

🚀 Getting Started
Prerequisites
Python 3.8+

The Enron Spam Dataset (enron_spam_data.csv) placed in the root directory.

Installation
Clone the repository:

git clone <your-repository-url>
cd <your-repository-directory>

Install the required Python packages:

pip install -r requirements.txt

(Note: You will need to create a requirements.txt file containing pandas, numpy, joblib, streamlit, scikit-learn, stable-baselines3, and gymnasium.)

Usage
Train the Model (First-time run):
The script is designed to automatically train and save the model if it doesn't find existing model files (spam_dqn_model.zip, tfidf_vectorizer.pkl, scaler.pkl). Simply run the Streamlit app, and it will trigger the training process.

Run the Streamlit App:
Execute the following command in your terminal:

streamlit run your_script_name.py

Open your web browser and navigate to the local URL provided by Streamlit (usually http://localhost:8501). Paste the email content into the text area and click "Check Spam" to get the classification.

📁 File Structure
.
├── enron_spam_data.csv      # The dataset used for training
├── main.py      # Main Python script with the Streamlit app
├── spam_dqn_model.zip       # Saved DQN model (generated after training)
├── tfidf_vectorizer.pkl     # Saved TF-IDF vectorizer (generated after training)
├── scaler.pkl               # Saved MinMaxScaler (generated after training)
└── README.md                # This README file
