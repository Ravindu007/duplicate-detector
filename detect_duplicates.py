import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import tensorflow as tf
import joblib
import requests
import string
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import os
import json
import io
import logging
import shutil
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress TensorFlow GPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Download necessary NLTK resources
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('omw-1.4', quiet=True)

# Create directory for models
os.makedirs('models', exist_ok=True)

# Load Google Service Account credentials
credentials_json = os.environ.get('GOOGLE_CREDENTIALS')
if not credentials_json:
    logging.error("GOOGLE_CREDENTIALS environment variable is not set or empty")
    exit(1)

try:
    json.loads(credentials_json)
except json.JSONDecodeError as e:
    logging.error(f"GOOGLE_CREDENTIALS contains invalid JSON: {e}")
    exit(1)

with open('credentials.json', 'w') as f:
    f.write(credentials_json)

# Authenticate with Google Drive API
try:
    creds = Credentials.from_service_account_file('credentials.json')
    drive_service = build('drive', 'v3', credentials=creds)
except Exception as e:
    logging.error(f"Error loading Google Service Account credentials: {e}")
    exit(1)

# File IDs
file_ids = {
    'best_traditional_model.h5': '1NlbKxkGoCpUiLINegN_cQL39RwAPQ5ji',
    'meta_model.joblib': '1S-o_KVIzs6O09uCNoc-VK9_v64DjQet3',
    'scaler.joblib': '1RHRI-nq6Poma2RyWfFYzTAV-TPRp2s4B',
    'tfidf_title.joblib': '1LJdAM4Iqu1GQKdXbSMjqXDQ4-OV54irG',
    'tfidf_body.joblib': '1TB0zQpMb6I4l3757H1HFwUs8vjhbMowf',
    'w2v_title.model': '1KrMXzTzvDjZHEh1VyDLgo3mSPu10FRJp',
    'w2v_body.model': '1E69ZbuMHofOILBgMGXQkr12ciSHLH6eR',
    'best_model_name.txt': '1gIUkSGur_B5ZPRsnAqvZyEtj5ru5SV9a',
    'codebert.zip': '1NmtxSf_xSdXOJUZ8i0AfMka2OkqMFVda'
}

# Download and validate files
for filename, file_id in file_ids.items():
    try:
        logging.info(f"Downloading {filename}...")
        request = drive_service.files().get_media(fileId=file_id)
        fh = io.FileIO(f'models/{filename}', 'wb')
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            logging.info(f"Download {int(status.progress() * 100)}%")
        fh.close()
        # Validate file size
        file_size = os.path.getsize(f'models/{filename}')
        if file_size == 0:
            raise ValueError(f"Downloaded file {filename} is empty")
        logging.info(f"Downloaded {filename} successfully, size: {file_size} bytes")
    except Exception as e:
        logging.error(f"Error downloading {filename}: {e}")
        exit(1)

# Unzip codebert.zip and adjust directory structure
try:
    logging.info("Unzipping codebert.zip...")
    os.system('unzip -o models/codebert.zip -d models/fine_tuned_codebert_model')
    if not os.path.exists('models/fine_tuned_codebert_model'):
        raise FileNotFoundError("Failed to unzip fine_tuned_codebert_model")
    # Move files from codebert subdirectory to fine_tuned_codebert_model
    codebert_subdir = os.path.join('models', 'fine_tuned_codebert_model', 'codebert')
    if os.path.exists(codebert_subdir):
        for file_name in os.listdir(codebert_subdir):
            src_path = os.path.join(codebert_subdir, file_name)
            dst_path = os.path.join('models', 'fine_tuned_codebert_model', file_name)
            shutil.move(src_path, dst_path)
        # Remove the empty codebert subdirectory
        os.rmdir(codebert_subdir)
    # Verify config.json exists
    if not os.path.exists('models/fine_tuned_codebert_model/config.json'):
        raise FileNotFoundError("config.json not found in models/fine_tuned_codebert_model")
except Exception as e:
    logging.error(f"Error unzipping codebert.zip or adjusting directory: {e}")
    exit(1)

# Load saved models and components
try:
    logging.info("Loading models...")
    with open('models/best_model_name.txt', 'r') as f:
        best_model_name = f.read().strip()
    best_traditional_model = tf.keras.models.load_model('models/best_traditional_model.h5')
    meta_model = joblib.load('models/meta_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    tfidf_title = joblib.load('models/tfidf_title.joblib')
    tfidf_body = joblib.load('models/tfidf_body.joblib')
    logging.info("Loading Word2Vec title model...")
    w2v_title = Word2Vec.load('models/w2v_title.model')
    logging.info("Loading Word2Vec body model...")
    w2v_body = Word2Vec.load('models/w2v_body.model')
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    logging.info("Loading fine-tuned CodeBERT model...")
    codebert_model = AutoModelForSequenceClassification.from_pretrained('models/fine_tuned_codebert_model')
except Exception as e:
    logging.error(f"Error loading models: {e}")
    exit(1)

# Preprocessing functions
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def full_preprocess(text):
    if not text or not isinstance(text, str):
        return ""
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def minimal_preprocess(text):
    if not text or not isinstance(text, str):
        return ""
    return BeautifulSoup(text, 'html.parser').get_text()

# Read issue data
try:
    with open(os.environ['GITHUB_EVENT_PATH'], 'r') as f:
        event_data = json.load(f)
    new_title = event_data['issue']['title'] or ""
    new_body = event_data['issue']['body'] or ""
except KeyError as e:
    logging.error(f"Error reading issue data: {e}")
    exit(1)

# Preprocess the new issue
new_title_clean = full_preprocess(new_title)
new_body_clean = full_preprocess(new_body)
new_title_raw = minimal_preprocess(new_title)
new_body_raw = minimal_preprocess(new_body)

# Fetch issues
repo = os.environ['GITHUB_REPOSITORY']
url = f"https://api.github.com/repos/{repo}/issues"
params = {"state": "all", "per_page": 10}
headers = {'Authorization': f'token {os.environ["GITHUB_TOKEN"]}'}
try:
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    issues = response.json()
except requests.RequestException as e:
    logging.error(f"Error fetching issues: {e}")
    exit(1)

# Compute features
def compute_features(pair_df):
    features = []
    for _, row in pair_df.iterrows():
        tfidf_t1 = tfidf_title.transform([row['Original Issue Title_clean']]).toarray()
        tfidf_t2 = tfidf_title.transform([row['Duplicate Issue Title_clean']]).toarray()
        tfidf_b1 = tfidf_body.transform([row['Original Issue Body_clean']]).toarray()
        tfidf_b2 = tfidf_body.transform([row['Duplicate Issue Body_clean']]).toarray()
        tfidf_title_sim = cosine_similarity(tfidf_t1, tfidf_t2)[0][0]
        tfidf_body_sim = cosine_similarity(tfidf_b1, tfidf_b2)[0][0]

        w2v_t1 = get_w2v_embedding(row['Original Issue Title_clean'].split(), w2v_title)
        w2v_t2 = get_w2v_embedding(row['Duplicate Issue Title_clean'].split(), w2v_title)
        w2v_b1 = get_w2v_embedding(row['Original Issue Body_clean'].split(), w2v_body)
        w2v_b2 = get_w2v_embedding(row['Duplicate Issue Body_clean'].split(), w2v_body)
        w2v_title_sim = cosine_similarity([w2v_t1], [w2v_t2])[0][0] if np.any(w2v_t1) and np.any(w2v_t2) else 0
        w2v_body_sim = cosine_similarity([w2v_b1], [w2v_b2])[0][0] if np.any(w2v_b1) and np.any(w2v_b2) else 0

        sbert_t1 = sbert_model.encode(row['Original Issue Title_clean'])
        sbert_t2 = sbert_model.encode(row['Duplicate Issue Title_clean'])
        sbert_b1 = sbert_model.encode(row['Original Issue Body_clean'])
        sbert_b2 = sbert_model.encode(row['Duplicate Issue Body_clean'])
        sbert_title_sim = cosine_similarity([sbert_t1], [sbert_t2])[0][0]
        sbert_body_sim = cosine_similarity([sbert_b1], [sbert_b2])[0][0]

        t1_tokens = set(row['Original Issue Title_clean'].split())
        t2_tokens = set(row['Duplicate Issue Title_clean'].split())
        b1_tokens = set(row['Original Issue Body_clean'].split())
        b2_tokens = set(row['Duplicate Issue Body_clean'].split())
        title_overlap = len(t1_tokens & t2_tokens) / len(t1_tokens | t2_tokens) if t1_tokens or t2_tokens else 0
        body_overlap = len(b1_tokens & b2_tokens) / len(b1_tokens | b2_tokens) if b1_tokens or b2_tokens else 0

        features.append([tfidf_title_sim, tfidf_body_sim, w2v_title_sim, w2v_body_sim,
                         sbert_title_sim, sbert_body_sim, title_overlap, body_overlap])
    return np.array(features)

def get_w2v_embedding(tokens, model):
    vectors = [model.wv[token] for token in tokens if token in model.wv]
    return np.mean(vectors, axis=0) if vectors else np.zeros(model.vector_size)

# Process issues
results = []
for issue in issues:
    fetched_title = issue.get('title', '')
    fetched_body = issue.get('body', '') or ''
    issue_number = issue.get('number', '')
    fetched_title_clean = full_preprocess(fetched_title)
    fetched_body_clean = full_preprocess(fetched_body)
    fetched_title_raw = minimal_preprocess(fetched_title)
    fetched_body_raw = minimal_preprocess(fetched_body)

    pair_df = pd.DataFrame({
        'Original Issue Title_clean': [new_title_clean],
        'Original Issue Body_clean': [new_body_clean],
        'Duplicate Issue Title_clean': [fetched_title_clean],
        'Duplicate Issue Body_clean': [fetched_body_clean],
        'Original Issue Title_raw': [new_title_raw],
        'Original Issue Body_raw': [new_body_raw],
        'Duplicate Issue Title_raw': [fetched_title_raw],
        'Duplicate Issue Body_raw': [fetched_body_raw]
    })

    try:
        pair_features = compute_features(pair_df)
        pair_features_scaled = scaler.transform(pair_features)
    except Exception as e:
        logging.error(f"Error computing features for issue #{issue_number}: {e}")
        continue

    try:
        if best_model_name == 'ANN':
            traditional_pred_prob = best_traditional_model.predict(pair_features_scaled, verbose=0).flatten()[0]
        elif best_model_name in ['LSTM', 'RNN']:
            pair_features_3d = pair_features_scaled.reshape(1, 1, pair_features_scaled.shape[1])
            traditional_pred_prob = best_traditional_model.predict(pair_features_3d, verbose=0).flatten()[0]
        elif best_model_name == 'CNN':
            pair_features_3d = pair_features_scaled.reshape(1, pair_features_scaled.shape[1], 1)
            traditional_pred_prob = best_traditional_model.predict(pair_features_3d, verbose=0).flatten()[0]
        else:
            raise ValueError(f"Unsupported model: {best_model_name}")
    except Exception as e:
        logging.error(f"Error predicting with traditional model for issue #{issue_number}: {e}")
        continue

    text = f"[CLS] {new_title_raw} [SEP] {new_body_raw} [SEP] {fetched_title_raw} [SEP] {fetched_body_raw}"
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    try:
        with torch.no_grad():
            outputs = codebert_model(**inputs)
            logits = outputs.logits
            pair_codebert_pred_prob = torch.softmax(logits, dim=1)[0, 1].item()
    except Exception as e:
        logging.error(f"Error predicting with CodeBERT for issue #{issue_number}: {e}")
        continue

    hybrid_features = np.array([[traditional_pred_prob, pair_codebert_pred_prob]])
    try:
        hybrid_pred_prob = meta_model.predict_proba(hybrid_features)[:, 1][0]
    except Exception as e:
        logging.error(f"Error predicting with meta-model for issue #{issue_number}: {e}")
        continue

    results.append({
        'issue_number': issue_number,
        'fetched_title': fetched_title,
        'fetched_body': fetched_body,
        'traditional_prob': traditional_pred_prob,
        'codebert_prob': pair_codebert_pred_prob,
        'hybrid_prob': hybrid_pred_prob
    })

# Save and output results
results_df = pd.DataFrame(results)
if not results_df.empty:
    results_df = results_df.sort_values(by='hybrid_prob', ascending=False)
    logging.info("New Issue:")
    logging.info(f"Title: {new_title}")
    logging.info(f"Body: {new_body}\n")
    logging.info("Top 5 potential duplicates (ranked by hybrid probability):")
    logging.info(results_df.head(5)[['issue_number', 'fetched_title', 'traditional_prob', 'codebert_prob', 'hybrid_prob']].to_string())
    results_df.to_csv('duplicate_ranking.csv', index=False)
    logging.info("\nFull ranking saved to 'duplicate_ranking.csv'")
else:
    logging.info("No duplicate issues detected.")
    pd.DataFrame().to_csv('duplicate_ranking.csv', index=False)
