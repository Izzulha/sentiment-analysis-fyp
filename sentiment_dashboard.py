# sentiment_dashboard.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import os
import base64
import re
import torch
import warnings
warnings.filterwarnings('ignore')


def get_base64_image(image_path):
    """Return a base64 image string, or an empty string if the image is missing."""
    try:
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode()
    except Exception:
        pass
    return ""

# ========== PAGE CONFIGURATION ==========
st.set_page_config(
    page_title="MigraStat Malaysia",
    page_icon="Logo.png",   # ← your logo
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== ENHANCED CUSTOM CSS ==========
st.markdown("""
<style>
    /* Hide the 'View Fullscreen' button on images */
    button[title="View fullscreen"] {
        display: none !important;
    }

    /* Target the specific Streamlit test ID for the fullscreen button */
    [data-testid="StyledFullScreenButton"] {
        display: none !important;
    }
    
    /* Sidebar Styling - Enhanced */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f9fafc 100%);
        border-right: 1px solid #e8eaf6;
        box-shadow: 2px 0 15px rgba(0, 0, 0, 0.05);
        padding-top: 0;
    }
    
    /* Sidebar Content Container */
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0 !important;
    }
    
    /* Logo Header Section - Reduced height */
    .sidebar-header {
        background: #FFFFFF !important;
        padding: 15px 20px 5px 20px;
        margin: -1rem -1rem 0.5rem -1rem;
        position: relative;
        overflow: hidden;
    }
    
    .logo-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 8px;
        margin-bottom: 5px;
    }
    
    .logo-icon {
        font-size: 24px;
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #7A46BE, #5d4de1);
        color: white;
        margin-bottom: 5px;
    }
    
    .logo-text {
        text-align: center;
    }
    
    .logo-text h1 {
        color: #1a237e;
        font-size: 16px;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.3px;
        line-height: 1.2;
    }
    
    .logo-text p {
        color: #5d6d7c;
        font-size: 10px;
        margin: 2px 0 0 0;
        font-weight: 400;
        letter-spacing: 0.3px;
    }
    
    /* Section Labels */
    .section-label {
        font-size: 10px;
        color: #90a4ae;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding: 15px 20px 8px 20px;
        margin: 0 -20px;
        background: #f8fafc;
        border-top: 1px solid #e8eaf6;
        border-bottom: 1px solid #e8eaf6;
    }
    
    /* Navigation Menu */
    .sidebar-nav {
        padding: 0;
    }
    
    /* Menu Items Styling - Enhanced */
    .menu-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 12px 20px;
        margin: 0;
        background: transparent;
        color: #5d6d7c;
        font-size: 13px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        text-decoration: none;
        border-left: 3px solid transparent;
        position: relative;
        overflow: hidden;
    }
    
    .menu-item:hover {
        background: linear-gradient(90deg, rgba(122, 70, 190, 0.08) 0%, rgba(122, 70, 190, 0.04) 100%);
        color: #7A46BE;
        transform: translateX(4px);
    }
    
    .menu-item.active {
        background: linear-gradient(90deg, rgba(122, 70, 190, 0.12) 0%, rgba(122, 70, 190, 0.06) 100%);
        color: #7A46BE;
        font-weight: 600;
        border-left: 4px solid #7A46BE;
    }
    
    .menu-item.active::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #7A46BE, #5d4de1);
    }
    
    .menu-icon {
        font-size: 16px;
        width: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
    }
    
    .menu-item:hover .menu-icon {
        transform: scale(1.1);
    }
    
    .menu-item.active .menu-icon {
        transform: scale(1.15);
    }
    
    .menu-text {
        flex: 1;
        letter-spacing: 0.2px;
    }
    
    .menu-badge {
        background: #7A46BE;
        color: white;
        font-size: 10px;
        padding: 2px 6px;
        border-radius: 10px;
        font-weight: 600;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.8; }
        100% { opacity: 1; }
    }
    
    /* Collapsible Sidebar Indicator */
    .collapse-indicator {
        position: absolute;
        top: 50%;
        right: 15px;
        transform: translateY(-50%);
        font-size: 10px;
        color: #90a4ae;
        transition: transform 0.3s ease;
    }
    
    .menu-item:hover .collapse-indicator {
        color: #7A46BE;
    }
    
    /* Compact Model Status */
    .model-status-compact {
        background: #ffffff;
        border: 1px solid #e8eaf6;
        border-radius: 8px;
        padding: 10px 12px;
        margin: 15px 15px 10px 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .status-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 6px;
    }
    
    .status-label {
        font-size: 10px;
        color: #5d6d7c;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-indicator {
        display: flex;
        align-items: center;
        gap: 4px;
        font-size: 10px;
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
    }
    
    .status-dot.active {
        background: #4caf50;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.2);
    }
    
    .status-dot.inactive {
        background: #f44336;
        box-shadow: 0 0 0 2px rgba(244, 67, 54, 0.2);
    }
    
    .status-info {
        font-size: 10px;
        color: #90a4ae;
        line-height: 1.3;
    }
    
    /* Footer Section - Compact */
    .sidebar-footer {
        padding: 15px 20px 20px 20px;
        margin-top: auto;
        text-align: center;
        border-top: 1px solid #f0f2f6;
    }
    
    .footer-text {
        color: #90a4ae;
        font-size: 9px;
        line-height: 1.4;
        margin-bottom: 8px;
    }
    
    .footer-version {
        background: rgba(122, 70, 190, 0.1);
        color: #7A46BE;
        font-size: 9px;
        padding: 3px 8px;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        letter-spacing: 0.5px;
    }
    
    /* Main content styling */
    .stApp {
        background-color: #F5F7FB;
    }
    
    /* Main header */
    .main-header {
        color: #1a237e;
        font-size: 28px;
        font-weight: 600;
        margin-bottom: 5px;
        letter-spacing: -0.5px;
    }
    
    /* Sub-header */
    .sub-header {
        color: #5d6d7c;
        font-size: 14px;
        font-weight: 400;
        margin-bottom: 30px;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e8eaf6;
        height: 130px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.05);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card1 {
        background: #7A46BE;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #7A46BE;
        height: 130px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
        position: relative;
        overflow: hidden;
    }
    
    div.metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: #FFFFFF !important;
    }
    
    .metric-value1 {
        font-size: 32px;
        font-weight: 600;
        color: #FFFFFF;
        margin: 10px 0 5px 0;
        line-height: 1;
    }            
    
    .metric-value {
        font-size: 32px;
        font-weight: 600;
        color: #1a237e;
        margin: 10px 0 5px 0;
        line-height: 1;
    }
    
    .metric-label1 {
        font-size: 13px;
        color: #FFFFFF;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
            
    .metric-label {
        font-size: 13px;
        color: #546e7a;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 8px;
    }
    
    .metric-subtitle {
        font-size: 12px;
        color: #90a4ae;
        font-weight: 400;
    }
    
    .metric-subtitle1 {
        font-size: 12px;
        color: #FFFFFF;
        font-weight: 400;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom colors */
    .green-text { color: #4caf50; }
    .negative-result .result-badge {
     background: #f44336;
     color: white !important;
     box-shadow: 0 2px 8px #f4433680;
}
    .blue-text { color: #2196f3; }
    
    /* Chart containers */
    .chart-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e8eaf6;
        margin-bottom: 20px;
        box-shadow: 0 1px 20px rgba(0,0,0,0.12);
    }
    
    .chart-title {
        font-size: 16px;
        font-weight: 600;
        color: #1a237e;
        margin-bottom: 15px;
        display: flex;
        justify-content: center;   /* horizontal center */
        align-items: center;       /* vertical center */
        text-align: center;
            
    }
    
    /* Comment list styling */
    .comment-card {
        background: #ffffff;
        border: 1px solid #e8eaf6;
        border-radius: 10px;
        padding: 16px 16px; ;
        margin-bottom: 6px;
        transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .comment-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    
    .sentiment-badge {
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        border: 1px solid;
    }
    
    /* Chart improvements */
    .plot-container {
        padding-top: 10px;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        [data-testid="stSidebar"] {
            min-width: 220px !important;
            max-width: 220px !important;
        }
        
        .menu-item {
            padding: 10px 15px;
            font-size: 12px;
        }
        
        .logo-text h1 {
            font-size: 14px;
        }
    }
    
    /* Custom styling for analysis results */
    .result-container {
        background: linear-gradient(135deg, #4caf5020 0%, #4caf5010 100%);
        border: 1px solid #4caf5040;
        border-radius: 12px;
        padding: 25px;
        margin-top: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .result-title {
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .result-emoji {
        font-size: 32px;
    }
    
    .result-text {
        font-size: 20px;
        font-weight: 600;
        color: #4caf50;
    }
    
    .result-badge {
        padding: 10px 20px;
        border-radius: 25px;
        background: #4caf50;
        color: white;
        font-weight: 600;
        font-size: 16px;
        box-shadow: 0 2px 8px #4caf5080;
    }
    
    .text-display {
        margin-top: 20px;
        padding: 15px;
        background: rgba(255,255,255,0.8);
        border-radius: 8px;
        border-left: 4px solid #4caf50;
    }
    
    .text-label {
        font-size: 14px;
        font-weight: 500;
        color: #37474f;
        margin-bottom: 8px;
    }
    
    .text-content {
        font-size: 14px;
        color: #546e7a;
        line-height: 1.5;
        padding: 10px;
        background: rgba(255,255,255,0.9);
        border-radius: 6px;
        max-height: 150px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    .analysis-details {
        margin-top: 20px;
        padding: 15px;
        background: rgba(255,255,255,0.8);
        border-radius: 8px;
    }
    
    .details-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 10px;
        font-size: 13px;
        color: #546e7a;
    }
    
    .detail-box {
        background: white;
        padding: 10px;
        border-radius: 6px;
        text-align: center;
    }
    
    .detail-value {
        font-weight: 600;
        color: #4caf50;
        font-size: 16px;
    }
    
    .negative-result .result-text,
    .negative-result .result-badge,
    .negative-result .detail-value,
    .negative-result .text-display {
        border-color: #f44336;
        color: #f44336;
        background-color: #f4433620;
    }
    
    .negative-result .result-badge {
        background: #f44336;
        box-shadow: 0 2px 8px #f4433680;
    }
    
    .negative-result {
        background: linear-gradient(135deg, #f4433620 0%, #f4433610 100%);
        border-color: #f4433640;
    }
    
    /* Change Streamlit primary button color to purple */
    button[kind="primary"] {
        background: linear-gradient(90deg, #7A46BE, #5d4de1) !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(122, 70, 190, 0.35) !important;
    }

    /* Hover effect */
    button[kind="primary"]:hover {
        background: linear-gradient(90deg, #6a3db2, #4c3ccf) !important;
        box-shadow: 0 6px 16px rgba(122, 70, 190, 0.45) !important;
    }

    /* Issue Card Styling */
    .issue-card {
        background: linear-gradient(135deg, #ffffff 0%, #f9fafc 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e8eaf6;
        margin-bottom: 20px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    .issue-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(122, 70, 190, 0.15);
        border-color: #7A46BE;
    }
    
    .issue-icon {
        font-size: 36px;
        margin-bottom: 15px;
        display: inline-block;
        width: 70px;
        height: 70px;
        line-height: 70px;
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(122, 70, 190, 0.1) 0%, rgba(93, 77, 225, 0.1) 100%);
        color: #7A46BE;
    }
    
    .issue-title {
        font-size: 18px;
        font-weight: 600;
        color: #1a237e;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 2px solid #e8eaf6;
    }
    
    .issue-text {
        font-size: 14px;
        color: #546e7a;
        line-height: 1.6;
        margin: 0;
    }
    
    /* Issue specific colors */
    .jobs-card {
        border-top: 4px solid #FF6B6B;
    }
    
    .social-card {
        border-top: 4px solid #4ECDC4;
    }
    
    .economic-card {
        border-top: 4px solid #45B7D1;
    }
    
    .jobs-card .issue-icon {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1) 0%, rgba(255, 140, 107, 0.1) 100%);
        color: #FF6B6B;
    }
    
    .social-card .issue-icon {
        background: linear-gradient(135deg, rgba(78, 205, 196, 0.1) 0%, rgba(78, 205, 196, 0.1) 100%);
        color: #4ECDC4;
    }
    
    .economic-card .issue-icon {
        background: linear-gradient(135deg, rgba(69, 183, 209, 0.1) 0%, rgba(69, 183, 209, 0.1) 100%);
        color: #45B7D1;
    }
    
    /* Word Cloud Styling */
    .wordcloud-container {
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #e8eaf6;
        margin-bottom: 20px;
        box-shadow: 0 1px 20px rgba(0,0,0,0.12);
    }
    
    .wordcloud-title {
        font-size: 16px;
        font-weight: 600;
        color: #1a237e;
        margin-bottom: 15px;
        text-align: center;
    }
    
    .wordcloud-image-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 10px;
    }
    
    .wordcloud-image {
        max-width: 100%;
        height: auto;
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

</style>
""", unsafe_allow_html=True)

# ========== BERT MODEL FUNCTIONS ==========
@st.cache_resource
def load_bert_model():
    """Load your trained BERT sentiment analysis model"""
    try:
        model_path = "./model_savee"
        
        if not os.path.exists(model_path):
            return None, None, None
        
        # Check for required BERT model files
        required_files = ['config.json', 'model.safetensors', 'vocab.txt']
        existing_files = os.listdir(model_path)
        
        # Check which required files exist
        missing_files = [f for f in required_files if f not in existing_files]
        
        if missing_files:
            return None, None, None
        
        # Import transformers
        try:
            from transformers import BertTokenizer, BertForSequenceClassification
        except ImportError:
            return None, None, None
        
        # Load tokenizer and model
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        return tokenizer, model, device
        
    except Exception:
        return None, None, None

def analyze_sentiment_bert(text, tokenizer, model, device):
    """Analyze sentiment using BERT model"""
    try:
        # Tokenize
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        # Move to device
        input_ids = encoded_dict['input_ids'].to(device)
        attention_mask = encoded_dict['attention_mask'].to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        
        # Get probabilities
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        # Get the highest probability label
        pred_label_idx = torch.argmax(probs, dim=1).cpu.numpy()[0]
        
        # Map to labels (0: Negative, 1: Positive)
        label_map = {0: 'Negative', 1: 'Positive'}
        sentiment = label_map[pred_label_idx]
        
        return sentiment
        
    except Exception:
        return None

# ========== RULE-BASED SENTIMENT ANALYSIS ==========
def analyze_sentiment_rule_based(text):
    """Fallback rule-based sentiment analysis for Malay/English"""
    # Clean text
    text = str(text).strip().lower()
    
    # Indonesian positive words
    indo_positive_words = [
        'baik', 'bagus', 'senang', 'puas', 'mantap', 'hebat', 'luar biasa',
        'terbaik', 'bagus sekali', 'sangat baik', 'memuaskan', 'keren',
        'wow', 'oke', 'ok', 'yes', 'ya', 'sukses', 'berhasil', 'happy',
        'gembira', 'bahagia', 'positif', 'membantu', 'bermanfaat',
        'terima kasih', 'thanks', 'thank you', 'suka', 'menyenangkan'
    ]
    
    # Indonesian negative words
    indo_negative_words = [
        'buruk', 'jelek', 'tidak baik', 'tidak bagus', 'kecewa', 'sedih',
        'mengecewakan', 'parah', 'jelek sekali', 'sangat buruk', 'gagal',
        'tidak suka', 'tidak senang', 'marah', 'kesal', 'frustasi',
        'negatif', 'susah', 'sulit', 'problematik', 'masalah', 'rusak',
        'error', 'gagal', 'payah', 'lemah', 'tidak puas'
    ]
    
    # English positive words
    english_positive_words = [
        'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
        'awesome', 'perfect', 'happy', 'satisfied', 'pleased', 'love',
        'like', 'best', 'positive', 'helpful', 'useful', 'successful',
        'brilliant', 'outstanding', 'superb', 'terrific', 'nice', 'fine'
    ]
    
    # English negative words
    english_negative_words = [
        'bad', 'poor', 'terrible', 'awful', 'horrible', 'disappointed',
        'sad', 'angry', 'upset', 'frustrated', 'hate', 'dislike',
        'worst', 'negative', 'problem', 'issue', 'difficult', 'hard',
        'complicated', 'wrong', 'failed', 'failure', 'unhappy'
    ]
    
    # Combine all word lists
    positive_words = indo_positive_words + english_positive_words
    negative_words = indo_negative_words + english_negative_words
    
    # Count occurrences
    text_lower = text.lower()
    pos_count = 0
    neg_count = 0
    
    for word in positive_words:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            pos_count += 1
    
    for word in negative_words:
        if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
            neg_count += 1
    
    # Special patterns for Indonesian
    indo_positive_patterns = [
        r'sangat\s+(baik|bagus|puas|senang|mantap)',
        r'(baik|bagus|keren|hebat)\s+sekali',
        r'aku\s+(suka|senang|bahagia|gembira)',
        r'saya\s+(suka|senang|bahagia|gembira)',
        r'terima\s+kasih',
        r'thank\s+you',
        r'makasih'
    ]
    
    indo_negative_patterns = [
        r'tidak\s+(baik|bagus|senang|suka|puas|mau|ingin)',
        r'sangat\s+(buruk|jelek|kecewa|sedih|parah)',
        r'(buruk|jelek|parah)\s+sekali',
        r'aku\s+(tidak|kecewa|sedih|marah|kesal)',
        r'saya\s+(tidak|kecewa|sedih|marah|kesal)',
        r'masalah\s+(besar|serius)',
        r'gagal\s+total'
    ]
    
    # Check patterns
    for pattern in indo_positive_patterns:
        if re.search(pattern, text_lower):
            pos_count += 2
    
    for pattern in indo_negative_patterns:
        if re.search(pattern, text_lower):
            neg_count += 2
    
    # Determine sentiment
    if pos_count > neg_count:
        sentiment = "Positive"
    else:
        sentiment = "Negative"
    
    return sentiment

# ========== MAIN SENTIMENT ANALYSIS FUNCTION ==========
def analyze_sentiment_custom(text):
    """Main sentiment analysis function"""
    # Try BERT model first
    if bert_tokenizer is not None and bert_model is not None and device is not None:
        sentiment = analyze_sentiment_bert(text, bert_tokenizer, bert_model, device)
        if sentiment is not None:
            return sentiment
    
    # Fallback to rule-based
    return analyze_sentiment_rule_based(text)

# ========== DATA LOADING FUNCTIONS ==========
@st.cache_data(ttl=300)
def load_sentiment_data():
    try:
        file_paths = [
            "sentiment_results3.xlsx",
            "./sentiment_results3.xlsx",
            "data/sentiment_results3.xlsx",
            "../sentiment_results3.xlsx"
        ]
        
        for file_path in file_paths:
            if os.path.exists(file_path):
                df = pd.read_excel(file_path)
                return df
        
        # Create sample data if file not found
        st.warning("📁 Using sample data - Place 'sentiment_results3.xlsx' in the same directory for your data")
        dates = pd.date_range(start='2024-01-01', end='2024-12-27', freq='D')
        n_samples = len(dates)
        
        sample_data = {
            'createTimeISO': dates,
            'label': np.random.choice(['Positive', 'Negative'], n_samples, p=[0.67, 0.33]),
            'text': [f"Sample comment {i}" for i in range(n_samples)],
        }
        
        df = pd.DataFrame(sample_data)
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        # Create minimal sample data
        dates = pd.date_range(start='2024-01-01', end='2024-12-27', freq='D')
        n_samples = 100
        
        sample_data = {
            'createTimeISO': np.random.choice(dates, n_samples),
            'label': np.random.choice(['Positive', 'Negative'], n_samples, p=[0.67, 0.33]),
            'text': [f"Sample comment {i}" for i in range(n_samples)],
        }
        
        return pd.DataFrame(sample_data)

# ========== FUNCTION TO CHECK AND LOAD IMAGES ==========
def check_and_load_wordcloud_images():
    """Check if word cloud images exist and return their paths"""
    image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
    pos_image = None
    neg_image = None
    
    # Check for positive word cloud image
    for ext in image_extensions:
        pos_path = f"pos{ext}"
        if os.path.exists(pos_path):
            pos_image = pos_path
            break
    
    # Check for negative word cloud image
    for ext in image_extensions:
        neg_path = f"pos1{ext}"
        if os.path.exists(neg_path):
            neg_image = neg_path
            break
    
    return pos_image, neg_image

# ========== PAGE CONTENT FUNCTIONS ==========
def render_homepage(df):
    """Render the homepage content"""
    # Calculate statistics
    total_comments = len(df)
    positive_comments = len(df[df['label'] == 'Positive'])
    negative_comments = len(df[df['label'] == 'Negative'])
    positive_percentage = (positive_comments / total_comments * 100) if total_comments > 0 else 0
    negative_percentage = (negative_comments / total_comments * 100) if total_comments > 0 else 0
    
    # Calculate average monthly values for y-axis formatting
    if len(df) > 0:
        monthly_counts = df.groupby(df['createTimeISO'].dt.to_period('M')).size()
        max_count = monthly_counts.max()
    else:
        max_count = 0
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown('<h1 class="main-header">My Immigrant Sentiment Insights</h1>', unsafe_allow_html=True)
        st.markdown('<p class="sub-header">Analyzing public comments and sentiment trends</p>', unsafe_allow_html=True)

    # ========== METRIC CARDS ==========
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card1">
            <div class="metric-label1">TOTAL COMMENTS</div>
            <div class="metric-value1">{total_comments:,}</div>
            <div class="metric-subtitle1">Public Comments</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">POSITIVE SENTIMENT</div>
            <div class="metric-value" style="color: #4caf50;">{positive_comments:,}</div>
            <div class="metric-subtitle">{positive_percentage:.1f}% of total</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">NEGATIVE SENTIMENT</div>
            <div class="metric-value" style="color: #f44336;">{negative_comments:,}</div>
            <div class="metric-subtitle">{negative_percentage:.1f}% of total</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ========== CHARTS SECTION ==========
    col_left, col_right = st.columns([7, 3])

    with col_left:
        # Create a container for the line chart
        with st.container():
            # Prepare monthly data
            monthly_data = (
               df.groupby(['month_num', 'month', 'label'])
              .size()
              .unstack(fill_value=0)
              .reset_index()
              .sort_values('month_num')
            )
            
            # Create line chart
            fig_line = go.Figure()
            
            # Format y-axis based on data range
            if max_count < 100:
                tickformat = ',d'
                yaxis_title = "Number of Comments"
            elif max_count < 10000:
                tickformat = ',d'
                yaxis_title = "Number of Comments"
            else:
                tickformat = '~s'
                yaxis_title = "Comments (in thousands)"
            
            if 'Positive' in monthly_data.columns:
                fig_line.add_trace(go.Scatter(
                    x=monthly_data['month'],
                    y=monthly_data['Positive'],
                    mode='lines+markers',
                    name='Positive',
                    line=dict(color='#00D49B', width=3),
                    marker=dict(size=8, color='white', line=dict(width=2, color='#00D49B')),
                    hovertemplate=(
                        '<b>%{x}</b><br>' +
                        '<span style="color:#00D49B">● Positive: %{y:,}</span><br>' +
                        '<extra></extra>'
                    )
                ))
            
            if 'Negative' in monthly_data.columns:
                fig_line.add_trace(go.Scatter(
                    x=monthly_data['month'],
                    y=monthly_data['Negative'],
                    mode='lines+markers',
                    name='Negative',
                    line=dict(color='#742CDF', width=3),
                    marker=dict(size=8, color='white', line=dict(width=2, color='#742CDF')),
                    hovertemplate=(
                        '<b>%{x}</b><br>' +
                        '<span style="color:#742CDF">● Negative: %{y:,}</span><br>' +
                        '<extra></extra>'
                    )
                ))
            
            fig_line.update_layout(
                title=dict(
                    text="<b>📈 Monthly Sentiment Analysis</b>",
                    x=0.02,
                    xanchor='left',
                    y=0.95,
                    font=dict(size=16, color='#1a237e')
                ),
                height=320,
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    showgrid=False,
                    tickfont=dict(size=10, color='#5d6d7c'),
                    title=None,
                    tickangle=-45,
                    tickmode='array',
                    tickvals=monthly_data['month'],
                    ticktext=[m.split()[0] for m in monthly_data['month']]
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='#f0f0f0',
                    gridwidth=1,
                    tickfont=dict(size=10, color='#5d6d7c'),
                    tickformat=tickformat,
                    title=dict(
                        text=yaxis_title,
                        font=dict(size=11, color='#5d6d7c')
                    ),
                    zeroline=True,
                    zerolinecolor='#e0e0e0',
                    zerolinewidth=1
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=11),
                    bgcolor='rgba(255, 255, 255, 0.9)',
                    bordercolor='#e8eaf6',
                    borderwidth=1
                ),
                margin=dict(l=50, r=20, t=80, b=60),
                hovermode='x unified',
                hoverlabel=dict(
                    bgcolor='white',
                    font_size=11,
                    font_family="Arial",
                    bordercolor='#e8eaf6'
                )
            )
            
            st.plotly_chart(fig_line, use_container_width=True, config={'displaylogo': False})

    with col_right:
        # Create a container for the donut chart
        with st.container():
            # Create enhanced donut chart
            fig_donut = go.Figure()
            
            # Add pie chart trace
            fig_donut.add_trace(go.Pie(
                labels=['Positive', 'Negative'],
                values=[positive_comments, negative_comments],
                hole=0.65,
                marker=dict(
                    colors=['#00D49B', '#742CDF'],
                    line=dict(color='white', width=2)
                ),
                textinfo='percent+label',
                textposition='outside',
                textfont=dict(
                    size=11,
                    color=['#00D49B', '#742CDF']
                ),
                hoverinfo='label+value+percent',
                hovertemplate=(
                    "<b>%{label}</b><br>" +
                    "Count: %{value:,}<br>" +
                    "Percentage: %{percent:.1%}<br>" +
                    "<extra></extra>"
                ),
                pull=[0.02, 0],
                rotation=45
            ))
            
            # Single centered annotation for better clarity
            fig_donut.update_layout(
                title=dict(
                    text="<b>🥧 Sentiment Distribution</b>",
                    x=0.02,
                    xanchor='left',
                    y=0.95,
                    font=dict(size=15, color='#1a237e')
                ),
                height=320,
                showlegend=False,
                margin=dict(t=70, b=50, l=20, r=20),
                paper_bgcolor='white',
                plot_bgcolor='white',
                annotations=[
                    dict(
                        text=f"<b style='font-size:20px; color:#1a237e'>{total_comments:,}</b><br>"
                             f"<span style='font-size:11px; color:#5d6d7c'>Total Comments</span><br><br>",
                        x=0.5,
                        y=0.5,
                        font=dict(
                            family="Arial",
                            size=12
                        ),
                        showarrow=False,
                        align="center"
                    )
                ],
                hoverlabel=dict(
                    bgcolor='white',
                    font_size=11,
                    font_family="Arial",
                    bordercolor='#e8eaf6'
                )
            )
            
            st.plotly_chart(fig_donut, use_container_width=True)
    
    # ========== WORD CLOUD DASHBOARD SECTION ==========
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<h3 class="main-header">📊 Word Cloud Analysis</h3>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Visual representation of most frequent words in comments</p>', unsafe_allow_html=True)
    
    # Check and load word cloud images
    pos_image_path, neg_image_path = check_and_load_wordcloud_images()
    
    with st.container():
     col_word1, col_word2 = st.columns(2)

    
    with col_word1:
        # Positive Word Cloud Container
        with st.container():
            st.markdown('<div class="wordcloud-title">Positive Sentiment Word Cloud</div>', unsafe_allow_html=True)
            
            if pos_image_path:
                st.markdown('<div class="wordcloud-image-container">', unsafe_allow_html=True)
                try:
                    st.image(pos_image_path, use_container_width=True, caption="Most frequent words in positive comments")
                except Exception as e:
                    st.error(f"Error loading positive word cloud image: {str(e)}")
                    st.info("Please ensure 'pos.png' exists in the same directory")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Positive word cloud image not found. Please add 'pos.png' to the directory.")
                st.info("Common words in positive comments: baik (good), bagus (great), senang (happy), suka (like), terima kasih (thank you)")
            
            # Positive sentiment insights
            st.markdown("""
            <div style="margin-top: 15px; padding: 15px; background: linear-gradient(135deg, #4caf5020 0%, #4caf5010 100%); 
                        border-radius: 8px; border-left: 4px solid #4caf50;">
                <h4 style="color: #1a237e; margin-bottom: 10px; font-size: 14px;">💡 Positive Sentiment Insights</h4>
                <ul style="color: #546e7a; font-size: 13px; padding-left: 20px; margin: 0;">
                    <li>Words like "baik" (good), "respect", and "good" show high satisfaction and mutual respect.</li>
                    <li>"Senang" (happy) and "rasa" (feel) indicate positive personal feelings.</li>
                    <li>Frequent use of "terima kasih" (thank you) reflects appreciation.</li>
                    <li>Terms like "better", "betul" (correct), and "everyone" suggest inclusive, positive progress.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col_word2:
        # Negative Word Cloud Container
        with st.container():
            st.markdown('<div class="wordcloud-title">Negative Sentiment Word Cloud</div>', unsafe_allow_html=True)
            
            if neg_image_path:
                st.markdown('<div class="wordcloud-image-container">', unsafe_allow_html=True)
                try:
                    st.image(neg_image_path, use_container_width=True, caption="Most frequent words in negative comments")
                except Exception as e:
                    st.error(f"Error loading negative word cloud image: {str(e)}")
                    st.info("Please ensure 'pos1.png' exists in the same directory")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Negative word cloud image not found. Please add 'pos1.png' to the directory.")
                st.info("Common words in negative comments: buruk (bad), jelek (ugly), kecewa (disappointed), masalah (problem), susah (difficult)")
            
            # Negative sentiment insights
            st.markdown("""
            <div style="margin-top: 15px; padding: 15px; background: linear-gradient(135deg, #f4433620 0%, #f4433610 100%); 
                        border-radius: 8px; border-left: 4px solid #f44336;">
                <h4 style="color: #1a237e; margin-bottom: 10px; font-size: 14px;">💡 Negative Sentiment Insights</h4>
                <ul style="color: #546e7a; font-size: 13px; padding-left: 20px; margin: 0;">
                    <li>Financial Stress: Heavy focus on "gaji" (salary), "bayar" (pay), and "cukai" (tax) highlights money-related grievances.</li>
                    <li>Words like "permit", "kerajaan" (government), and "asing" (foreign) point to administrative or legal frustration.</li>
                    <li>"Salah" (wrong), "masalah" (problem), and "susah" (difficult) signal specific obstacles.</li>
                    <li>Frequent use of "tidak" (no/not) and "gagal" (failed) confirms negative outcomes.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

def render_comment_list(df):
    """Render the comment list page with pagination (30 comments per page)"""
    st.markdown('<h1 class="main-header">💬 Comment List</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Browse and search through all comments with sentiment analysis</p>', unsafe_allow_html=True)
    
    # Initialize session state for pagination
    if 'page_number' not in st.session_state:
        st.session_state.page_number = 1
    
    if 'comment_filter_state' not in st.session_state:
        st.session_state.comment_filter_state = {}
    
    # Search and filter section
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_query = st.text_input("🔍 Search comments...", "")
    
    with col2:
        sentiment_filter = st.selectbox(
            "Filter by sentiment",
            ["All", "Positive", "Negative"]
        )
    
    with col3:
        date_filter = st.selectbox(
            "Time period",
            ["All time", "Last 7 days", "Last 30 days", "Last 90 days"]
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    # Search filter
    if search_query:
        filtered_df = filtered_df[
            filtered_df['text'].str.contains(search_query, case=False, na=False)
        ]
    
    # Sentiment filter
    if sentiment_filter != "All":
        filtered_df = filtered_df[filtered_df['label'] == sentiment_filter]
    
    # Date filter
    if date_filter != "All time":
        today = pd.Timestamp.today()
        if date_filter == "Last 7 days":
            cutoff_date = today - pd.Timedelta(days=7)
        elif date_filter == "Last 30 days":
            cutoff_date = today - pd.Timedelta(days=30)
        elif date_filter == "Last 90 days":
            cutoff_date = today - pd.Timedelta(days=90)
        
        filtered_df = filtered_df[filtered_df['createTimeISO'] >= cutoff_date]
    
    # Reset page number if filters have changed
    current_filter_state = {
        'search': search_query,
        'sentiment': sentiment_filter,
        'date': date_filter,
        'count': len(filtered_df)
    }
    
    if (st.session_state.comment_filter_state != current_filter_state):
        st.session_state.page_number = 1
        st.session_state.comment_filter_state = current_filter_state
    
    if len(filtered_df) == 0:
        st.info("No comments found matching your filters.")
        st.session_state.page_number = 1
    else:
        # Apply default sorting (Newest first)
        filtered_df = filtered_df.sort_values('createTimeISO', ascending=False)
        
        # Calculate pagination
        total_comments = len(filtered_df)
        comments_per_page = 30
        total_pages = (total_comments + comments_per_page - 1) // comments_per_page
        
        # Ensure page number is within valid range
        if st.session_state.page_number < 1:
            st.session_state.page_number = 1
        elif st.session_state.page_number > total_pages:
            st.session_state.page_number = total_pages
        
        # Calculate start and end indices
        start_idx = (st.session_state.page_number - 1) * comments_per_page
        end_idx = min(start_idx + comments_per_page, total_comments)
        
        # Display comments for current page
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Show page info
        st.markdown(f"""
        <div style="background: #ffffff; border-radius: 8px; padding: 15px; margin-bottom: 20px; border: 1px solid #e8eaf6;">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    Showing comments <strong>{start_idx + 1}-{end_idx}</strong> of <strong>{total_comments:,}</strong> total comments
                </div>
                <div>
                    Page <strong>{st.session_state.page_number}</strong> of <strong>{total_pages}</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Display comments for current page
        for idx in range(start_idx, end_idx):
            row = filtered_df.iloc[idx]
            sentiment = row['label']
            sentiment_color = "#4caf50" if sentiment == "Positive" else "#f44336"
            sentiment_bg = "#e8f5e9" if sentiment == "Positive" else "#fdecea"
            comment_date = row['createTimeISO'].strftime("%b %d, %Y at %I:%M %p")
            
            # Display exact text from Excel (no truncation)
            comment_text = row['text']
            
            st.markdown(f"""
            <div class="comment-card">
                <div style="display: flex; justify-content: space-between; align-items: center; gap: 10px;">
                    <div style="flex: 1;">
                        <div style="font-size: 14px; line-height: 1.2; color: #263238; margin-bottom: 4px; ">
                            {comment_text}
                        </div>
                        <div style="font-size: 12px; color: #90a4ae;">
                            {comment_date}
                        </div>
                    </div>
                    <div style="margin-left: 4px;">
                        <div class="sentiment-badge" style="color: {sentiment_color}; background: {sentiment_bg}; border-color: {sentiment_color};">
                            {sentiment}
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        # Quick page navigation
        st.markdown("<br>", unsafe_allow_html=True)
        col_nav1, col_nav2, col_nav3 = st.columns([1, 2, 1])
        
        with col_nav2:
            page_numbers = []
            # Show first 3 pages
            for i in range(1, min(4, total_pages + 1)):
                page_numbers.append(i)
            
            # Show current page and surrounding pages
            if st.session_state.page_number > 3:
                page_numbers.append("...")
            
            start_page = max(1, st.session_state.page_number - 1)
            end_page = min(total_pages, st.session_state.page_number + 1)
            for i in range(start_page, end_page + 1):
                if i not in page_numbers:
                    page_numbers.append(i)
            
            # Show last 3 pages
            if total_pages > 3:
                for i in range(max(total_pages - 2, end_page + 1), total_pages + 1):
                    if i not in page_numbers:
                        page_numbers.append(i)
            
            # Create page buttons
            cols = st.columns(len(page_numbers))
            for i, page_btn in enumerate(page_numbers):
                with cols[i]:
                    if isinstance(page_btn, int):
                        if page_btn == st.session_state.page_number:
                            st.markdown(f"""
                            <div style="background: #7A46BE; color: white; padding: 8px; border-radius: 5px; 
                                        text-align: center; font-weight: bold;">
                                {page_btn}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            if st.button(str(page_btn), key=f"page_{page_btn}", use_container_width=True):
                                st.session_state.page_number = page_btn
                                st.rerun()
                    else:
                        st.markdown(f"""
                        <div style="padding: 8px; text-align: center; color: #90a4ae;">
                            {page_btn}
                        </div>
                        """, unsafe_allow_html=True)
        
        # Display total pages info
        st.markdown(f"""
        <div style="text-align: center; color: #546e7a; font-size: 14px; margin-top: 15px;">
            Page {st.session_state.page_number} of {total_pages} • {total_comments:,} total comments • {comments_per_page} per page
        </div>
        """, unsafe_allow_html=True)


def render_custom_text_analysis():
    """Render the custom text analysis page"""
    st.markdown('<h1 class="main-header">📝 Insert Your Own Words</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Analyze custom text for sentiment using our BERT model</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create container for the entire analysis section
        with st.container():
            st.markdown("""
            <div class="chart-container">
                <div class="chart-title">
                    Text Analysis
                </div>
            """, unsafe_allow_html=True)
            
            # Text input
            user_text = st.text_area(
                "Enter text to analyze:",
                height=150,
                placeholder="Type or paste your text here.",
                help="Enter any text you want to analyze for sentiment. Supports both Malay and English.",
                key="user_text_input"
            )
            
            # Clear button
            col_btn1, col_btn2 = st.columns([1, 1])
            with col_btn1:
                analyze_clicked = st.button("🤖 Analyze with BERT", type="primary", use_container_width=True, key="analyze_btn")
            with col_btn2:
                if st.button("🗑️ Clear Text", use_container_width=True, key="clear_btn"):
                    st.session_state.user_text_input = ""
                    st.rerun()
            
            # Analysis results
            if analyze_clicked and user_text.strip():
                with st.spinner("BERT model analyzing..."):
                    # Use BERT model for analysis
                    sentiment = analyze_sentiment_custom(user_text)
                    
                    # Display result
                    if sentiment == "Positive":
                        icon = "✅"
                        title = "Positive Sentiment"
                        emoji = "😊"
                        result_class = ""
                    else:
                        icon = "⚠️"
                        title = "Negative Sentiment"
                        emoji = "😞"
                        result_class = "negative-result"
                    
                    # Create a clean result display
                    st.markdown(f"""
                    <div class="result-container {result_class}">
                        <div class="result-header">
                            <div class="result-title">
                                <div class="result-emoji">{emoji}</div>
                                <div>
                                    <div class="result-text">
                                        {icon} {title}
                                    </div>
                                </div>
                            </div>
                            <div class="result-badge">
                                {sentiment}
                            </div>
                        </div>
                        
                    """, unsafe_allow_html=True)
                    
                    # Store in session for batch analysis
                    if 'analyzed_texts' not in st.session_state:
                        st.session_state.analyzed_texts = []
                    
                    st.session_state.analyzed_texts.append({
                        'text': user_text[:80] + ('...' if len(user_text) > 80 else ''),
                        'sentiment': sentiment,
                        'timestamp': datetime.now().strftime("%H:%M:%S"),
                        'model': 'BERT' if bert_model is not None else 'Rule-Based'
                    })
                    
            elif analyze_clicked and not user_text.strip():
                st.warning("⚠️ Please enter some text to analyze.")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Show recent analyses
        if 'analyzed_texts' in st.session_state and st.session_state.analyzed_texts:
            with st.container():
                st.markdown("""
                <div class="chart-container" style="margin-top: 20px;">
                    <div class="chart-title">
                        Recent Analyses
                    </div>
                """, unsafe_allow_html=True)
                
                # Display recent analyses
                for item in st.session_state.analyzed_texts[-5:]:
                    sent_color = "#4caf50" if item['sentiment'] == "Positive" else "#f44336"
                    with st.container():
                        cols = st.columns([4, 1, 1])
                        with cols[0]:
                            st.markdown(
                              f"<div style='margin-left: 15px;'>{item['text']}</div>",
                             unsafe_allow_html=True
                            )
                        with cols[1]:
                            st.markdown(f"<div style='color: {sent_color}; font-weight: bold;'>{item['sentiment']}</div>", unsafe_allow_html=True)
                        with cols[2]:
                            st.text(item['timestamp'])
                
                if st.button("Clear History", type="secondary", key="clear_history"):
                    st.session_state.analyzed_texts = []
                    st.rerun()
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # How to Use section in a container
        with st.container():
            how_to_html = """
            <div class="chart-container">
                <div class="chart-title">
                    How to Use
                </div>
                <div style="font-size: 14px; line-height: 1.6; color: #546e7a;">
                    <p><strong>📝 1.</strong> Type or paste your text</p>
                    <p><strong>💻 2.</strong> Click "Analyze with BERT"</p>
                    <p><strong>📊 3.</strong> View sentiment results</p>
                    <br>
                    <p><strong>💡 About BERT Model:</strong></p>
                    <ul style="padding-left: 20px;">
                        <li>State-of-the-art NLP model</li>
                        <li>Trained on immigrant sentiment data</li>
                        <li>Understands context & nuance</li>
                        <li>Results: Positive or Negative only</li>
                    </ul>
                </div>
            </div>
            """
            st.markdown(how_to_html, unsafe_allow_html=True)

def render_model_comparison():
    """Render the model comparison page - UPDATED TO VERTICAL LAYOUT"""
    st.markdown('<h1 class="main-header">📊 Model Comparison</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Comparing performance of Random Forest, SVM, and mBERT models</p>', unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # ========== THREE BAR CHARTS FOR EACH MODEL (HORIZONTAL) ==========
    st.markdown("""
    <div class="chart-container">
        <div class="chart-title">
            Model Improvement Through Hyperparameter Tuning
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    # Color definitions for split ratios
    color_6040 = "#301CA0"  # 60/40 split
    color_7030 = "#7132CA"  # 70/30 split
    color_8020 = "#C47BE4"  # 80/20 split
    
    # Split ratio colors mapping
    split_colors = {
        '60/40': color_6040,
        '70/30': color_7030,
        '80/20': color_8020
    }
    
    # Random Forest Bar Chart
    with col1:
        # Create container for Random Forest chart
        with st.container():
            # Data from the images
            split_ratios = ['60/40', '70/30', '80/20']
            
            # Random Forest data
            rf_pre = [61.38, 62.38, 62.17]
            rf_tune1 = [70.90, 70.49, 70.30]
            rf_tune2 = [74.78, 74.19, 74.40]
            rf_tune3 = [74.05, 74.95, 75.47]
            
            stages = ['Pre', 'Tuning 1', 'Tuning 2', 'Tuning 3']
            
            fig_rf = go.Figure()
            
            for i, split in enumerate(split_ratios):
                values = [rf_pre[i], rf_tune1[i], rf_tune2[i], rf_tune3[i]]
                fig_rf.add_trace(go.Bar(
                    name=split,
                    x=stages,
                    y=values,
                    text=[f'{v:.1f}%' for v in values],
                    textposition='auto',
                    marker=dict(color=split_colors[split]),
                    hovertemplate=f'Split: {split}<br>Stage: %{{x}}<br>Accuracy: %{{y:.2f}}%<extra></extra>'
                ))
            
            fig_rf.update_layout(
                title=dict(
                    text="<b>🌲 Random Forest Tuning Progress</b>",
                    x=0.02,
                    xanchor='left',
                    y=0.95,
                    font=dict(size=16, color='#1a237e')
                ),
                height=350,
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    title='Tuning Stage',
                    showgrid=False,
                    tickfont=dict(size=11),
                    title_font=dict(size=12)
                ),
                yaxis=dict(
                    title='Accuracy (%)',
                    range=[55, 80],
                    showgrid=True,
                    gridcolor='#f0f0f0',
                    tickfont=dict(size=11),
                    title_font=dict(size=12)
                ),
                barmode='group',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10),
                    title=dict(
                        text="Train/Test Split",
                        font=dict(size=11)
                    )
                ),
                hoverlabel=dict(
                    bgcolor='white',
                    font_size=11,
                    font_family="Arial"
                ),
                margin=dict(l=50, r=20, t=80, b=50)
            )
            
            st.plotly_chart(fig_rf, use_container_width=True, config={'displaylogo': False})
            
            # RF Insight
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; border-left: 4px solid {color_8020}; margin-top: 10px;">
                <p style="margin: 0; color: #546e7a; font-size: 12px;">
                    <strong>🌲 Random Forest:</strong> Shows steady improvement from 61.38% to 75.47% (80/20 split). 
                    Tuning 2 provides best overall improvement.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # SVM Bar Chart
    with col2:
        # Create container for SVM chart
        with st.container():
            # SVM data
            svm_pre = [54.76, 54.81, 54.79]
            svm_tune1 = [51.39, 51.58, 51.83]
            svm_tune2 = [63.48, 64.78, 66.46]
            svm_tune3 = [71.00, 71.96, 72.51]
            
            fig_svm = go.Figure()
            
            for i, split in enumerate(split_ratios):
                values = [svm_pre[i], svm_tune1[i], svm_tune2[i], svm_tune3[i]]
                fig_svm.add_trace(go.Bar(
                    name=split,
                    x=stages,
                    y=values,
                    text=[f'{v:.1f}%' for v in values],
                    textposition='auto',
                    marker=dict(color=split_colors[split]),
                    hovertemplate=f'Split: {split}<br>Stage: %{{x}}<br>Accuracy: %{{y:.2f}}%<extra></extra>'
                ))
            
            fig_svm.update_layout(
                title=dict(
                    text="<b>⚡ SVM Tuning Progress</b>",
                    x=0.02,
                    xanchor='left',
                    y=0.95,
                    font=dict(size=16, color='#1a237e')
                ),
                height=350,
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    title='Tuning Stage',
                    showgrid=False,
                    tickfont=dict(size=11),
                    title_font=dict(size=12)
                ),
                yaxis=dict(
                    title='Accuracy (%)',
                    range=[40, 80],
                    showgrid=True,
                    gridcolor='#f0f0f0',
                    tickfont=dict(size=11),
                    title_font=dict(size=12)
                ),
                barmode='group',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10),
                    title=dict(
                        text="Train/Test Split",
                        font=dict(size=11)
                    )
                ),
                hoverlabel=dict(
                    bgcolor='white',
                    font_size=11,
                    font_family="Arial"
                ),
                margin=dict(l=50, r=20, t=80, b=50)
            )
            
            st.plotly_chart(fig_svm, use_container_width=True, config={'displaylogo': False})
            
            # SVM Insight
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; border-left: 4px solid {color_8020}; margin-top: 10px;">
                <p style="margin: 0; color: #546e7a; font-size: 12px;">
                    <strong>⚡ SVM:</strong> Initial drop in Tuning 1, then significant improvement to 72.51%. 
                    Most sensitive to hyperparameter changes.
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # mBERT Bar Chart - UPDATED WITH ALL 4 STAGES
    with col3:
        # Create container for mBERT chart
        with st.container():
            # mBERT data - Updated with all 4 stages from both images
            # First image values for Pre and Tuning 1 (81.9%, 89.2%, 89.6%, 90.2%)
            # Second image table values for Tuning 2 and Tuning 3
            mbert_pre = [81.9, 81.9, 81.9]  # Same for all splits as per first image
            mbert_tune1 = [89.2, 89.2, 89.2]  # Same for all splits as per first image
            
            # Second image values - three rows for each split ratio
            # Assuming: Row1=60/40, Row2=70/30, Row3=80/20
            mbert_tune2 = [89.43, 90.53, 89.56]   # From second image
            mbert_tune3 = [85.73, 87.73, 88.37]   # From second image
            
            mbert_stages = ['Pre', 'Tuning 1', 'Tuning 2', 'Tuning 3']
            
            fig_mbert = go.Figure()
            
            for i, split in enumerate(split_ratios):
                values = [mbert_pre[i], mbert_tune1[i], mbert_tune2[i], mbert_tune3[i]]
                fig_mbert.add_trace(go.Bar(
                    name=split,
                    x=mbert_stages,
                    y=values,
                    text=[f'{v:.1f}%' for v in values],
                    textposition='auto',
                    marker=dict(color=split_colors[split]),
                    hovertemplate=f'Split: {split}<br>Stage: %{{x}}<br>Accuracy: %{{y:.2f}}%<extra></extra>'
                ))
            
            fig_mbert.update_layout(
                title=dict(
                    text="<b>🤖 mBERT Tuning Progress</b>",
                    x=0.02,
                    xanchor='left',
                    y=0.95,
                    font=dict(size=16, color='#1a237e')
                ),
                height=350,
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    title='Tuning Stage',
                    showgrid=False,
                    tickfont=dict(size=11),
                    title_font=dict(size=12)
                ),
                yaxis=dict(
                    title='Accuracy (%)',
                    range=[75, 95],
                    showgrid=True,
                    gridcolor='#f0f0f0',
                    tickfont=dict(size=11),
                    title_font=dict(size=12)
                ),
                barmode='group',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=10),
                    title=dict(
                        text="Train/Test Split",
                        font=dict(size=11)
                    )
                ),
                hoverlabel=dict(
                    bgcolor='white',
                    font_size=11,
                    font_family="Arial"
                ),
                margin=dict(l=50, r=20, t=80, b=50)
            )
            
            st.plotly_chart(fig_mbert, use_container_width=True, config={'displaylogo': False})
            
            # mBERT Insight - Updated
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; border-left: 4px solid {color_8020}; margin-top: 10px;">
                <p style="margin: 0; color: #546e7a; font-size: 12px;">
                    <strong>🤖 mBERT:</strong> High baseline (81.9%). Tuning 1 provides significant jump to 89.2%. 
                    Additional tuning shows variations with best performance in Tuning 2 for 70/30 split (90.53%).
                </p>
            </div>
            """, unsafe_allow_html=True)
    
    # ========== FINAL COMPARISON SECTION - UPDATED TO VERTICAL LAYOUT ==========
    st.markdown("<br>", unsafe_allow_html=True)
    
    # BOTTOM SECTION: Performance Summary (Full Width)
    with st.container():
        st.markdown("""
        <div class="chart-container">
            <div class="chart-title">
                Performance Summary
            </div>
            <div style="padding: 20px;">
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                    <div style="background: linear-gradient(135deg, #301CA010 0%, #301CA005 100%); padding: 20px; border-radius: 8px; border-left: 4px solid #301CA0;">
                        <h4 style="color: #2c3e50; margin: 0 0 15px 0; text-align: center;">🌲 Random Forest</h4>
                        <div style="text-align: center; margin-bottom: 15px;">
                            <div style="font-size: 32px; font-weight: 700; color: #301CA0;">76.04%</div>
                            <div style="font-size: 12px; color: #546e7a;">Best Accuracy (80/20 Split)</div>
                        </div>
                        <p style="color: #546e7a; margin: 0; font-size: 13px; line-height: 1.6;">
                            <strong>Optimal Split:</strong> 80% Train / 20% Test<br>
                            <strong>Improvement:</strong> +13.30% from baseline<br>
                            <strong>Strength:</strong> Handles non-linear patterns well
                        </p>
                    </div>
                    <div style="background: linear-gradient(135deg, #7132CA10 0%, #7132CA05 100%); padding: 20px; border-radius: 8px; border-left: 4px solid #7132CA;">
                        <h4 style="color: #2c3e50; margin: 0 0 15px 0; text-align: center;">⚡ SVM</h4>
                        <div style="text-align: center; margin-bottom: 15px;">
                            <div style="font-size: 32px; font-weight: 700; color: #7132CA;">74.02%</div>
                            <div style="font-size: 12px; color: #546e7a;">Best Accuracy (80/20 Split)</div>
                        </div>
                        <p style="color: #546e7a; margin: 0; font-size: 13px; line-height: 1.6;">
                            <strong>Optimal Split:</strong> 80% Train / 20% Test<br>
                            <strong>Improvement:</strong> +17.72% from baseline<br>
                            <strong>Note:</strong> Most sensitive to parameter tuning
                        </p>
                    </div>
                    <div style="background: linear-gradient(135deg, #C47BE410 0%, #C47BE405 100%); padding: 20px; border-radius: 8px; border-left: 4px solid #C47BE4;">
                        <h4 style="color: #2c3e50; margin: 0 0 15px 0; text-align: center;">🤖 mBERT</h4>
                        <div style="text-align: center; margin-bottom: 15px;">
                            <div style="font-size: 32px; font-weight: 700; color: #C47BE4;">90.53%</div>
                            <div style="font-size: 12px; color: #546e7a;">Best Accuracy (70/30 Split, Tuning 2)</div>
                        </div>
                        <p style="color: #546e7a; margin: 0; font-size: 13px; line-height: 1.6;">
                            <strong>Optimal Split:</strong> 70% Train / 30% Test<br>
                            <strong>Improvement:</strong> +8.63% from baseline<br>
                            <strong>Strength:</strong> State-of-the-art for text classification
                        </p>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # ========== METHODOLOGY SECTION ==========
    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.container():
        st.markdown("""
        <div class="chart-container">
            <div class="chart-title">
                Methodology & Analysis Details
            </div>
            <div style="padding: 20px;">
                <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
                    <div>
                        <h4 style="color: #2c3e50; margin-bottom: 10px;">📊 Evaluation Metrics</h4>
                        <ul style="color: #546e7a; padding-left: 20px;">
                            <li><strong>Primary Metric:</strong> Accuracy (%)</li>
                            <li><strong>Dataset:</strong> 8000 cleaned labeled comments</li>
                            <li><strong>Languages:</strong> Malay & English</li>
                            <li><strong>Validation:</strong> Three split ratios tested</li>
                        </ul>
                    </div>
                    <div>
                        <h4 style="color: #2c3e50; margin-bottom: 10px;">🔧 Tuning Process</h4>
                        <ul style="color: #546e7a; padding-left: 20px;">
                            <li><strong>Random Forest:</strong> n_estimators, max_depth, min_samples_split</li>
                            <li><strong>SVM:</strong> C, kernel, gamma parameters</li>
                            <li><strong>mBERT:</strong> Learning rate, batch size, epochs</li>
                            <li><strong>All models:</strong> Grid search with cross-validation</li>
                        </ul>
                    </div>
                </div>
                <div style="margin-top: 20px; padding: 15px; background: linear-gradient(135deg, #301CA010 0%, #7132CA10 50%, #C47BE410 100%); border-radius: 8px; border-left: 4px solid #7A46BE;">
                    <h4 style="color: #2c3e50; margin-bottom: 10px;">🎯 Key Insights</h4>
                    <p style="color: #546e7a; margin: 0; font-size: 14px; line-height: 1.6;">
                        <strong>Split Ratio Impact:</strong> All models perform best with 80/20 train/test split, suggesting 
                        that more training data significantly improves performance for sentiment analysis tasks.<br><br>
                        <strong>mBERT Superiority:</strong> The mBERT model significantly outperforms traditional machine 
                        learning models, achieving 90.53% accuracy with tuning. This demonstrates the power of 
                        transformer-based models for multilingual text classification tasks.<br><br>
                        <strong>Traditional ML Limitations:</strong> While Random Forest and SVM show improvement with 
                        hyperparameter tuning, they plateau around 75% accuracy, highlighting the limitations of 
                        traditional approaches for complex NLP tasks.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
def render_about():
    """Render the about page with ISSUE section in right column"""
    st.markdown('<h1 class="main-header">ℹ️ About</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Learn more about this sentiment analysis dashboard</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # About section in container
        with st.container():
            about_html = """
            <div class="chart-container">
                <div style="padding: 24px; text-align: justify; text-indent: 1em;">
                    <h3 style="margin-top: 0;">📊 My Immigrant Sentiment Insights Dashboard</h3>
                    <p class="about-text">
                        This dashboard is designed to analyze and visualize public sentiment toward immigrant workers in Malaysia using 
                        artificial intelligence and natural language processing techniques. Public opinions are gathered from social media platforms,
                        where discussions about immigration often reflect diverse economic, social, and employment-related perspectives. 
                        By analyzing these comments, the system helps reveal overall sentiment patterns and public attitudes surrounding immigrant workers.
                    </p>
                    <p class="about-text">
                       The sentiment analysis is performed using a combination of machine learning and deep learning models,
                       with Multilingual BERT (mBERT) as the primary model. This approach enables the system to accurately process both English and Malay text,
                       which is essential in Malaysia's multilingual online environment. The integration of advanced NLP techniques ensures reliable sentiment 
                       classification even for informal and mixed-language content commonly found on social media.
                    </p>
                    <p class="about-text">
                        Through interactive visualizations, this dashboard allows users to explore sentiment distributions,
                        monthly trends, and individual comments in an intuitive way.
                        The goal is to transform large volumes of unstructured text data into meaningful insights that are easy to understand. Overall, this dashboard supports data-driven analysis and helps encourage informed discussions and evidence-based decision-making related to immigrant workers in Malaysia.
                    </p>
                </div>
            </div>
            """
            st.markdown(about_html, unsafe_allow_html=True)
        
        # Technical Details Section in container
        with st.container():
            st.markdown("""
            <div class="chart-container" style="margin-top: 20px;">
                <div class="chart-title">
                    Technical Details
                </div>
                <div style="padding: 20px;">
                    <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
                        <div>
                            <h4 style="color: #2c3e50; margin-bottom: 10px;">🔧 Technologies Used</h4>
                            <ul style="color: #546e7a; padding-left: 20px; margin: 0;">
                                <li>Python 3.9+</li>
                                <li>Streamlit for frontend</li>
                                <li>Transformers (Hugging Face)</li>
                                <li>Plotly for visualizations</li>
                                <li>Pandas and rapid miner for data processing</li>
                            </ul>
                        </div>
                        <div>
                            <h4 style="color: #2c3e50; margin-bottom: 10px;">📁 Data Sources</h4>
                            <ul style="color: #546e7a; padding-left: 20px; margin: 0;">
                                <li>X comments</li>
                                <li>Tiktok comments</li>
                            </ul>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        # ISSUE Section in Right Column in container
        with st.container():
            st.markdown("""
            <div class="chart-container">
                <div class="chart-title">
                    📌 Issues related to immigrant workers are frequently debated by the public.
                </div>
                <div style="padding: 10px;">
            """, unsafe_allow_html=True)
            
            # Job Issue Card
            st.markdown("""
            <div class="issue-card jobs-card">
                <div class="issue-icon">💼</div>
                <div class="issue-title">Jobs</div>
                <div class="issue-text">
                    Immigrants are debated for competing with locals for jobs, 
                    creating discussions about employment opportunities and market dynamics.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Social Integration Issue Card
            st.markdown("""
            <div class="issue-card social-card">
                <div class="issue-icon">👥</div>
                <div class="issue-title">Social Integration</div>
                <div class="issue-text">
                    People debate how well immigrants adapt to local culture and society, 
                    examining challenges and successes in multicultural integration.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Economic Impact Issue Card
            st.markdown("""
            <div class="issue-card economic-card">
                <div class="issue-icon">📈</div>
                <div class="issue-title">Economic Impact</div>
                <div class="issue-text">
                    Some see immigrants as boosting the economy, while others worry about 
                    costs and money sent abroad, creating balanced economic discussions.
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Contact & Support Section (Full width below columns)
    with st.container():
        st.markdown("""
        <div class="chart-container" style="margin-top: 20px;">
            <div class="chart-title">
                Contact & Support
            </div>
            <div style="padding: 20px;">
                <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px;">
                    <div>
                        <h4 style="color: #2c3e50; margin-bottom: 10px;">📧 Contact Information</h4>
                        <p style="color: #546e7a;">
                            For support or inquiries:<br>
                            <strong>Email:</strong> support@migrstat.com<br>
                            <strong>Website:</strong> www.migrstat.com<br>
                            <strong>Phone:</strong> +6012 565 1519
                        </p>
                    </div>
                    <div>
                        <h4 style="color: #2c3e50; margin-bottom: 10px;">🔄 Version Information</h4>
                        <p style="color: #546e7a;">
                            <strong>Current Version:</strong> 3.0.1<br>
                            <strong>Last Updated:</strong> December 2025<br>
                        </p>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    

# ========== ENHANCED SIDEBAR FUNCTION ==========
def render_sidebar():
    """Render the enhanced sidebar content with icons and branding"""
    with st.sidebar:
        # Branding Header with Logo - More compact
        st.markdown("""
        <div class="sidebar-header">
            <div class="logo-container">
                <img src="data:image/png;base64,{}"
                    style="width: 120px; margin-bottom: 5px;" />
                <div class="logo-text">
                    <p>Immigrant Sentiment Insights</p>
                </div>
            </div>
        </div>
        """.format(
            get_base64_image("Logo.png")
        ), unsafe_allow_html=True)
        
        # Navigation Menu
        if 'active_page' not in st.session_state:
            st.session_state.active_page = "home"
        
        st.markdown('<div class="sidebar-nav">', unsafe_allow_html=True)
        
        # Main Navigation Section
        st.markdown('<div class="section-label">Main Navigation</div>', unsafe_allow_html=True)
        
        # Primary Actions (Dashboard & Comment List)
        menu_items_primary = [
            {"icon": "📊", "label": "Dashboard", "page": "home"},
            {"icon": "💬", "label": "Comment List", "page": "comments"},
        ]
        
        for item in menu_items_primary:
            is_active = st.session_state.active_page == item["page"]
            
            if st.button(
                f"{item['icon']} {item['label']}",
                key=f"menu_{item['page']}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.active_page = item["page"]
                st.rerun()
        
        # Analysis Tools Section
        st.markdown('<div class="section-label">Analysis Tools</div>', unsafe_allow_html=True)
        
        menu_items_analysis = [
            {"icon": "🔍", "label": "Text Analysis", "page": "analysis"},
            {"icon": "📈", "label": "Model Comparison", "page": "comparison"},
        ]
        
        for item in menu_items_analysis:
            is_active = st.session_state.active_page == item["page"]
            
            if st.button(
                f"{item['icon']} {item['label']}",
                key=f"menu_{item['page']}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.active_page = item["page"]
                st.rerun()
        
        # About Section
        st.markdown('<div class="section-label">Information</div>', unsafe_allow_html=True)
        
        menu_items_info = [
            {"icon": "ℹ️", "label": "About", "page": "about"},
        ]
        
        for item in menu_items_info:
            is_active = st.session_state.active_page == item["page"]
            
            if st.button(
                f"{item['icon']} {item['label']}",
                key=f"menu_{item['page']}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                st.session_state.active_page = item["page"]
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Compact Model Status Indicator
        model_status = "Active" if bert_model is not None else "Inactive"
        model_type = "BERT Model" if bert_model is not None else "Rule-Based"
        status_color = "#4caf50" if bert_model is not None else "#f44336"
        status_dot_class = "active" if bert_model is not None else "inactive"
        
        st.markdown(f"""
        <div class="model-status-compact">
            <div class="status-header">
                <div class="status-label">Model Status</div>
                <div class="status-indicator">
                    <div class="status-dot {status_dot_class}"></div>
                    <span style="color: {status_color}; font-weight: 600;">{model_status}</span>
                </div>
            </div>
            <div class="status-info">
                Using: {model_type}<br>
                Language: Malay & English
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Footer Section - More compact
        st.markdown("""
        <div class="sidebar-footer">
            <div class="footer-text">
                Real-time sentiment analysis platform<br>
                Powered by AI & NLP
            </div>
            <div class="footer-version">v3.0.1</div>
        </div>
        """, unsafe_allow_html=True)

# ========== MAIN APPLICATION FLOW ==========
# Load BERT model
bert_tokenizer, bert_model, device = load_bert_model()

# Load data
df = load_sentiment_data()

# Process data
try:
    df['createTimeISO'] = pd.to_datetime(df['createTimeISO'])
    df['date'] = df['createTimeISO'].dt.date
    df['month'] = df['createTimeISO'].dt.strftime('%b %Y')
    df['hour'] = df['createTimeISO'].dt.hour
    df['day_name'] = df['createTimeISO'].dt.day_name()
    df['label'] = df['label'].str.strip().str.title()
    
    # Ensure only Positive or Negative
    df['label'] = df['label'].apply(lambda x: 'Positive' if 'positive' in str(x).lower() else 'Negative' if 'negative' in str(x).lower() else 'Negative')

    df['month_short'] = df['createTimeISO'].dt.strftime('%b')
    df['month_num'] = df['createTimeISO'].dt.month
    
except Exception as e:
    st.error(f"❌ Error processing data: {str(e)}")
    st.stop()

# Render sidebar
render_sidebar()

# Render main content based on active page
if st.session_state.active_page == "home":
    render_homepage(df)
elif st.session_state.active_page == "comments":
    render_comment_list(df)
elif st.session_state.active_page == "analysis":
    render_custom_text_analysis()
elif st.session_state.active_page == "comparison":
    render_model_comparison()
elif st.session_state.active_page == "about":
    render_about()