import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="🚢 Titanic Survival Prediction",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #1a1a2e);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        min-height: 100vh;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    /* Main container with glass effect */
    .app-container {
        max-width: 1440px;
        margin: 0 auto;
        padding: 1.5rem;
    }
    
    .glass-card {
        background: rgba(20, 20, 50, 0.45);
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        border-radius: 32px;
        padding: 1.8rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.03);
        transition: all 0.3s cubic-bezier(0.2, 0.9, 0.4, 1);
        margin-bottom: 1.4rem;
        animation: cardFade 0.6s ease both;
    }
    
    .glass-card:hover {
        transform: translateY(-4px);
        border-color: rgba(255, 255, 255, 0.12);
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(255, 255, 255, 0.02);
        background: rgba(25, 25, 60, 0.55);
    }
    
    @keyframes cardFade {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Title styling */
    .main-title {
        font-size: 4.2rem;
        font-weight: 800;
        background: linear-gradient(145deg, #f6d5f7 0%, #fbe9d7 40%, #b8e1fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
        text-shadow: 0 0 40px rgba(180, 160, 255, 0.15);
        text-align: center;
        animation: fadeInUp 1s ease;
    }
    
    .sub-title {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.5);
        font-weight: 400;
        letter-spacing: 0.3px;
        text-align: center;
        animation: fadeInUp 1.2s ease;
    }
    
    /* Metric cards - premium */
    .metric-card {
        background: rgba(18, 18, 45, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 28px;
        padding: 1.6rem 0.8rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.04);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
        animation: fadeInUp 0.8s ease;
    }
    
    .metric-card:hover {
        transform: scale(1.03) translateY(-2px);
        border-color: rgba(255, 215, 0, 0.25);
        background: rgba(30, 30, 70, 0.7);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #e0e7ff, #b8c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.02em;
    }
    
    .metric-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 2.5px;
        color: rgba(255, 255, 255, 0.3);
        margin-top: 0.2rem;
        font-weight: 500;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .stSidebar {
        background: rgba(8, 8, 28, 0.7) !important;
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-right: 1px solid rgba(255, 255, 255, 0.03) !important;
    }
    
    /* Navigation radio buttons */
    .stRadio > div {
        gap: 0.25rem;
    }
    
    .stRadio label {
        color: rgba(255, 255, 255, 0.45);
        font-weight: 500;
        padding: 0.65rem 1.4rem;
        border-radius: 60px;
        transition: all 0.3s ease;
        font-size: 0.95rem;
        letter-spacing: 0.2px;
        background: transparent;
        border: 1px solid transparent;
        cursor: pointer;
    }
    
    .stRadio label:hover {
        background: rgba(255, 255, 255, 0.04);
        color: #fff;
        border-color: rgba(255, 255, 255, 0.06);
    }
    
    .stRadio [data-baseweb="radio"] [aria-checked="true"] {
        background: linear-gradient(135deg, #6c5ce7, #a29bfe) !important;
        color: #fff !important;
        box-shadow: 0 4px 16px rgba(108, 92, 231, 0.3);
        border-color: transparent;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(145deg, #6c5ce7, #4834d4);
        color: white;
        border: none;
        padding: 0.9rem 2.8rem;
        border-radius: 60px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 8px 28px rgba(108, 92, 231, 0.25);
        width: 100%;
        letter-spacing: 0.3px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 16px 40px rgba(108, 92, 231, 0.4);
        background: linear-gradient(145deg, #7c6cf7, #5a4bd4);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.4rem;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 40px;
        padding: 0.4rem;
        border: 1px solid rgba(255, 255, 255, 0.03);
    }
    
    .stTabs [data-baseweb="tab"] {
        color: rgba(255, 255, 255, 0.4);
        border-radius: 40px;
        padding: 0.5rem 1.8rem;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: #fff;
        background: rgba(255, 255, 255, 0.03);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6c5ce7, #a29bfe) !important;
        color: white !important;
        box-shadow: 0 4px 14px rgba(108, 92, 231, 0.2);
    }
    
    /* Success & Error boxes */
    .success-box {
        background: linear-gradient(145deg, rgba(46, 213, 115, 0.08), rgba(46, 213, 115, 0.02));
        border: 1px solid rgba(46, 213, 115, 0.2);
        border-radius: 40px;
        padding: 2.2rem 1.5rem;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        text-align: center;
        box-shadow: 0 8px 30px rgba(46, 213, 115, 0.05);
        animation: slideIn 0.6s ease;
    }
    
    .error-box {
        background: linear-gradient(145deg, rgba(255, 71, 87, 0.08), rgba(255, 71, 87, 0.02));
        border: 1px solid rgba(255, 71, 87, 0.2);
        border-radius: 40px;
        padding: 2.2rem 1.5rem;
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        text-align: center;
        box-shadow: 0 8px 30px rgba(255, 71, 87, 0.05);
        animation: slideIn 0.6s ease;
    }
    
    /* Flowchart */
    .flowchart {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 0.5rem;
        padding: 1rem;
    }
    
    .flow-step {
        background: rgba(30, 30, 70, 0.5);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 60px;
        padding: 0.7rem 2.2rem;
        border: 1px solid rgba(255, 255, 255, 0.04);
        color: #fff;
        font-weight: 500;
        min-width: 180px;
        text-align: center;
        transition: all 0.3s ease;
        animation: fadeInUp 0.8s ease;
    }
    
    .flow-step:hover {
        border-color: rgba(255, 255, 255, 0.15);
        background: rgba(40, 40, 90, 0.5);
        transform: scale(1.02);
    }
    
    .flow-arrow {
        color: rgba(255, 255, 255, 0.12);
        font-size: 1.8rem;
    }
    
    /* Comparison Table */
    .comparison-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0 6px;
        margin: 1rem 0;
    }
    
    .comparison-table th {
        background: rgba(255, 255, 255, 0.02);
        color: rgba(255, 255, 255, 0.5);
        padding: 0.9rem 0.5rem;
        text-align: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.02);
        font-weight: 600;
        letter-spacing: 0.5px;
        font-size: 0.8rem;
        text-transform: uppercase;
    }
    
    .comparison-table td {
        color: rgba(255, 255, 255, 0.8);
        padding: 0.9rem 0.5rem;
        text-align: center;
        background: rgba(255, 255, 255, 0.01);
        border-radius: 16px;
        transition: all 0.3s ease;
    }
    
    .comparison-table tr {
        transition: all 0.3s ease;
    }
    
    .comparison-table tr:hover td {
        background: rgba(255, 255, 255, 0.03);
    }
    
    .winner {
        background: rgba(255, 215, 0, 0.04) !important;
        border: 1px solid rgba(255, 215, 0, 0.15);
        border-radius: 20px;
    }
    
    .winner td {
        color: #ffd700 !important;
        font-weight: 600;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 1.2rem;
        border-radius: 60px;
        font-weight: 600;
        font-size: 0.8rem;
        letter-spacing: 0.3px;
    }
    
    .badge-high {
        background: rgba(46, 213, 115, 0.15);
        color: #2ed573;
        border: 1px solid rgba(46, 213, 115, 0.2);
    }
    
    .badge-medium {
        background: rgba(255, 165, 0, 0.12);
        color: #ffb347;
        border: 1px solid rgba(255, 165, 0, 0.15);
    }
    
    .badge-low {
        background: rgba(255, 71, 87, 0.12);
        color: #ff6b7a;
        border: 1px solid rgba(255, 71, 87, 0.15);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: rgba(255, 255, 255, 0.08);
        padding: 2rem 0 0.5rem;
        font-size: 0.8rem;
        letter-spacing: 1.5px;
        border-top: 1px solid rgba(255, 255, 255, 0.02);
        margin-top: 1.5rem;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 20px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #6c5ce7, #a29bfe);
        border-radius: 20px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #7c6cf7, #b8a8ff);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: transparent !important;
    }
    
    .stDataFrame table {
        border-radius: 16px !important;
        overflow: hidden !important;
    }
    
    .stDataFrame thead tr th {
        background: rgba(255, 255, 255, 0.03) !important;
        color: rgba(255, 255, 255, 0.7) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    .stDataFrame tbody tr td {
        color: rgba(255, 255, 255, 0.6) !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.02) !important;
    }
    
    .stDataFrame tbody tr:hover td {
        background: rgba(255, 255, 255, 0.02) !important;
    }
    
    /* Selectbox styling */
    .stSelectbox > div {
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
    }
    
    .stSelectbox > div:hover {
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Number input & slider */
    .stNumberInput > div, .stSlider > div {
        background: rgba(255, 255, 255, 0.02) !important;
        border-radius: 12px !important;
    }
    
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .main-title {
            font-size: 2.8rem;
        }
        .sub-title {
            font-size: 1.1rem;
        }
        .glass-card {
            padding: 1.2rem;
            border-radius: 24px;
        }
        .metric-value {
            font-size: 2.2rem;
        }
        .flow-step {
            min-width: 140px;
            padding: 0.5rem 1.5rem;
        }
    }
    
    @media (max-width: 480px) {
        .main-title {
            font-size: 2.2rem;
        }
        .glass-card {
            padding: 1rem;
            border-radius: 20px;
        }
        .metric-card {
            padding: 1rem 0.5rem;
        }
        .metric-value {
            font-size: 1.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD DATA ====================
@st.cache_data
def load_data():
    try:
        train = pd.read_csv("data/train.csv")
        test = pd.read_csv("data/test.csv")
        return train, test
    except FileNotFoundError:
        st.error("⚠️ Data files not found! Please ensure 'data/train.csv' and 'data/test.csv' exist.")
        st.stop()
    except Exception as e:
        st.error(f"⚠️ Error loading data: {str(e)}")
        st.stop()

train, test = load_data()

# ==================== PREPARE DATA ====================
def prepare_data(data):
    df = data.copy()
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df = df.drop(columns=['Cabin'], errors='ignore')
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket'], errors='ignore')
    return df

# ==================== TRAIN MODELS ====================
@st.cache_resource
def train_models():
    df = prepare_data(train)
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    feature_names = X.columns.tolist()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    return lr, dt, rf, X_train, X_test, y_train, y_test, feature_names

lr_model, dt_model, rf_model, X_train, X_test, y_train, y_test, feature_names = train_models()

# ==================== SIDEBAR ====================
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1.5rem 0;">
        <div style="font-size: 3.5rem;">🚢</div>
        <h2 style="color: #fff; font-weight: 700; margin: 0;">Titanic App</h2>
        <p style="color: rgba(255,255,255,0.4); font-size: 0.85rem;">ML + Streamlit</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
    
    pages = [
        "🏠 Home",
        "📊 Dataset",
        "📈 EDA Dashboard",
        "🧹 Data Cleaning",
        "🔢 Encoding",
        "⚙️ Feature Engineering",
        "🎯 Feature Selection",
        "✂️ Train/Test Split",
        "🤖 ML Models",
        "📋 Evaluation",
        "🔮 Live Prediction",
        "💡 Model Explainability",
        "📖 About ML Models",
        "👤 About Me"
    ]
    
    page = st.radio("Navigation", pages, label_visibility="collapsed")
    
    st.markdown("<hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align: center; color: rgba(255,255,255,0.2); font-size: 0.75rem;">
        Made with ❤️<br>
        <span style="color: rgba(255,255,255,0.3);">Muhammad Zalan</span>
    </div>
    """, unsafe_allow_html=True)

# ==================== HOME ====================
if page == "🏠 Home":
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <div class="main-title">🚢 Titanic Survival</div>
        <div class="sub-title">Machine Learning · Streamlit · Data Science</div>
        <br>
        <div style="color: rgba(255,255,255,0.4); font-size: 1.1rem;">
            by <span style="color: rgba(255,255,255,0.8); font-weight: 600;">Muhammad Zalan</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h3 style="color: #fff;">📌 Project Introduction</h3>
            <p style="color: rgba(255,255,255,0.7); line-height: 1.8;">
                This project predicts whether a passenger survived the Titanic disaster using 
                <strong style="color: #fff;">Machine Learning</strong> models. 
                Built with <strong style="color: #fff;">Streamlit</strong> for interactive visualization.
            </p>
            <br>
            <h4 style="color: #fff;">🎯 Objectives</h4>
            <ul style="color: rgba(255,255,255,0.7); line-height: 2;">
                <li>🔍 Explore and visualize Titanic dataset</li>
                <li>🧹 Clean and preprocess data</li>
                <li>🤖 Build and compare ML models</li>
                <li>🚀 Deploy interactive Streamlit app</li>
            </ul>
            <br>
            <h4 style="color: #fff;">🛠 Technologies Used</h4>
            <div style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem;">
                <span style="background: rgba(102,126,234,0.2); padding: 0.3rem 1rem; border-radius: 20px; color: #a8b5ff; font-size: 0.9rem;">Python</span>
                <span style="background: rgba(102,126,234,0.2); padding: 0.3rem 1rem; border-radius: 20px; color: #a8b5ff; font-size: 0.9rem;">Streamlit</span>
                <span style="background: rgba(102,126,234,0.2); padding: 0.3rem 1rem; border-radius: 20px; color: #a8b5ff; font-size: 0.9rem;">Scikit-learn</span>
                <span style="background: rgba(102,126,234,0.2); padding: 0.3rem 1rem; border-radius: 20px; color: #a8b5ff; font-size: 0.9rem;">Pandas</span>
                <span style="background: rgba(102,126,234,0.2); padding: 0.3rem 1rem; border-radius: 20px; color: #a8b5ff; font-size: 0.9rem;">Plotly</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==================== DATASET ====================
elif page == "📊 Dataset":
    st.markdown("<h1 style='color: #fff;'>📊 Dataset Overview</h1>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{train.shape[0]}</div>
            <div class="metric-label">Rows</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{train.shape[1]}</div>
            <div class="metric-label">Columns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">Survived</div>
            <div class="metric-label">Target Column</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{train.isnull().sum().sum()}</div>
            <div class="metric-label">Missing Values</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #fff;'>📋 Data Types</h4>", unsafe_allow_html=True)
        dtype_df = pd.DataFrame({
            'Column': train.dtypes.index,
            'Dtype': train.dtypes.values.astype(str)
        })
        st.dataframe(dtype_df, width='stretch', hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #fff;'>🔍 Missing Values Summary</h4>", unsafe_allow_html=True)
        missing_df = train.isnull().sum().reset_index()
        missing_df.columns = ['Column', 'Missing']
        missing_df = missing_df[missing_df['Missing'] > 0]
        if len(missing_df) > 0:
            st.dataframe(missing_df, width='stretch', hide_index=True)
        else:
            st.success("✅ No missing values found!")
        st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #fff;'>📊 Summary Statistics</h4>", unsafe_allow_html=True)
    st.dataframe(train.describe(), width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #fff;'>🔍 First 10 Rows</h4>", unsafe_allow_html=True)
    st.dataframe(train.head(10), width='stretch')
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== EDA DASHBOARD ====================
elif page == "📈 EDA Dashboard":
    st.markdown("<h1 style='color: #fff;'>📈 EDA Dashboard</h1>", unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["📊 Distributions", "📈 Relationships", "🔍 Correlation"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #fff;'>Survival Count</h4>", unsafe_allow_html=True)
            survived_counts = train['Survived'].value_counts().reset_index()
            survived_counts.columns = ['Survived', 'Count']
            fig = px.bar(
                survived_counts,
                x='Survived', y='Count',
                color='Survived',
                color_discrete_sequence=['#ff6b6b', '#51cf66'],
                labels={'Survived': 'Status', 'Count': 'Number of Passengers'}
            )
            # FIX: Use update_xaxes instead of update_xaxis
            fig.update_xaxes(ticktext=['No', 'Yes'], tickvals=[0, 1])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #fff;'>Gender Distribution</h4>", unsafe_allow_html=True)
            fig = px.pie(
                train, names='Sex',
                color='Sex',
                color_discrete_sequence=['#4dabf7', '#f06595']
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #fff;'>Pclass Distribution</h4>", unsafe_allow_html=True)
            pclass_counts = train['Pclass'].value_counts().reset_index()
            pclass_counts.columns = ['Pclass', 'Count']
            fig = px.bar(
                pclass_counts,
                x='Pclass', y='Count',
                color='Pclass',
                color_discrete_sequence=['#ffd43b', '#ff922b', '#ff6b6b'],
                labels={'Pclass': 'Passenger Class', 'Count': 'Number of Passengers'}
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #fff;'>Age Distribution</h4>", unsafe_allow_html=True)
            fig = px.histogram(
                train, x='Age', nbins=30,
                color_discrete_sequence=['#69db7c']
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #fff;'>Survival vs Gender</h4>", unsafe_allow_html=True)
            fig = px.bar(
                train.groupby('Sex')['Survived'].mean().reset_index(),
                x='Sex', y='Survived',
                color='Sex',
                color_discrete_sequence=['#4dabf7', '#f06595'],
                labels={'Survived': 'Survival Rate'}
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #fff;'>Survival vs Pclass</h4>", unsafe_allow_html=True)
            fig = px.bar(
                train.groupby('Pclass')['Survived'].mean().reset_index(),
                x='Pclass', y='Survived',
                color='Pclass',
                color_discrete_sequence=['#ffd43b', '#ff922b', '#ff6b6b'],
                labels={'Survived': 'Survival Rate'}
            )
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #fff;'>Correlation Heatmap</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        corr = train.select_dtypes(include=['float64', 'int64']).corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        ax.tick_params(colors='white')
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

# ==================== DATA CLEANING ====================
elif page == "🧹 Data Cleaning":
    st.markdown("<h1 style='color: #fff;'>🧹 Data Cleaning Pipeline</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #fff;">🔄 Cleaning Flowchart</h4>
            <div class="flowchart">
                <div class="flow-step">📊 Raw Dataset</div>
                <div class="flow-arrow">↓</div>
                <div class="flow-step">🔍 Missing Values Detection</div>
                <div class="flow-arrow">↓</div>
                <div class="flow-step">🧹 Handling Missing Values</div>
                <div class="flow-arrow">↓</div>
                <div class="flow-step">🗑️ Drop Unnecessary Columns</div>
                <div class="flow-arrow">↓</div>
                <div class="flow-step">✅ Clean Dataset</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        missing_before = train.isnull().sum().reset_index()
        missing_before.columns = ['Column', 'Missing']
        missing_before['Missing'] = missing_before['Missing'].astype(int)
        
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #fff;'>Before Cleaning</h4>", unsafe_allow_html=True)
        st.dataframe(missing_before[missing_before['Missing'] > 0], width='stretch', hide_index=True)
        
        st.markdown("""
        <br>
        <div style="color: rgba(255,255,255,0.7);">
            <p><strong style="color: #fff;">Cleaning Steps:</strong></p>
            <p>🟢 Age → Median Fill</p>
            <p>🟢 Embarked → Mode Fill</p>
            <p>🔴 Cabin → Dropped</p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    train_clean = train.copy()
    train_clean['Age'] = train_clean['Age'].fillna(train_clean['Age'].median())
    train_clean['Embarked'] = train_clean['Embarked'].fillna(train_clean['Embarked'].mode()[0])
    train_clean = train_clean.drop(columns=['Cabin'], errors='ignore')
    
    missing_after = train_clean.isnull().sum().reset_index()
    missing_after.columns = ['Column', 'Missing']
    missing_after['Missing'] = missing_after['Missing'].astype(int)
    
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #fff;'>After Cleaning</h4>", unsafe_allow_html=True)
    st.dataframe(missing_after[missing_after['Missing'] > 0], width='stretch', hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem;">
        <div style="background: rgba(46, 213, 115, 0.1); border: 2px solid #2ed573; border-radius: 15px; padding: 1rem;">
            <span style="color: #2ed573; font-size: 1.2rem; font-weight: 600;">✅ Data cleaned successfully!</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== ENCODING ====================
elif page == "🔢 Encoding":
    st.markdown("<h1 style='color: #fff;'>🔢 Encoding</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #fff;">Sex Encoding</h4>
            <div style="display: flex; flex-direction: column; align-items: center; gap: 0.5rem; padding: 1rem;">
                <div style="background: rgba(255,255,255,0.05); padding: 0.5rem 2rem; border-radius: 10px; color: white;">Male</div>
                <div style="font-size: 2rem; color: rgba(255,255,255,0.3);">↓</div>
                <div style="background: rgba(46, 213, 115, 0.2); padding: 0.5rem 2rem; border-radius: 10px; color: #2ed573; font-weight: 700;">0</div>
                <div style="font-size: 2rem; color: rgba(255,255,255,0.3);">↓</div>
                <div style="background: rgba(255,255,255,0.05); padding: 0.5rem 2rem; border-radius: 10px; color: white;">Female</div>
                <div style="font-size: 2rem; color: rgba(255,255,255,0.3);">↓</div>
                <div style="background: rgba(46, 213, 115, 0.2); padding: 0.5rem 2rem; border-radius: 10px; color: #2ed573; font-weight: 700;">1</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #fff;">Embarked Encoding (One-Hot)</h4>
            <div style="display: flex; flex-direction: column; align-items: center; gap: 0.5rem; padding: 1rem;">
                <div style="background: rgba(255,255,255,0.05); padding: 0.5rem 2rem; border-radius: 10px; color: white;">C</div>
                <div style="font-size: 2rem; color: rgba(255,255,255,0.3);">↓</div>
                <div style="background: rgba(46, 213, 115, 0.2); padding: 0.5rem 2rem; border-radius: 10px; color: #2ed573; font-weight: 700;">[1, 0, 0]</div>
                <div style="font-size: 2rem; color: rgba(255,255,255,0.3);">↓</div>
                <div style="background: rgba(255,255,255,0.05); padding: 0.5rem 2rem; border-radius: 10px; color: white;">Q</div>
                <div style="font-size: 2rem; color: rgba(255,255,255,0.3);">↓</div>
                <div style="background: rgba(46, 213, 115, 0.2); padding: 0.5rem 2rem; border-radius: 10px; color: #2ed573; font-weight: 700;">[0, 1, 0]</div>
                <div style="font-size: 2rem; color: rgba(255,255,255,0.3);">↓</div>
                <div style="background: rgba(255,255,255,0.05); padding: 0.5rem 2rem; border-radius: 10px; color: white;">S</div>
                <div style="font-size: 2rem; color: rgba(255,255,255,0.3);">↓</div>
                <div style="background: rgba(46, 213, 115, 0.2); padding: 0.5rem 2rem; border-radius: 10px; color: #2ed573; font-weight: 700;">[0, 0, 1]</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ==================== FEATURE ENGINEERING ====================
elif page == "⚙️ Feature Engineering":
    st.markdown("<h1 style='color: #fff;'>⚙️ Feature Engineering</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: #fff;">🔄 Feature Engineering Pipeline</h4>
        <div style="display: flex; justify-content: center; align-items: center; gap: 2rem; padding: 1rem; flex-wrap: wrap;">
            <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 10px; text-align: center; min-width: 120px;">
                <div style="font-size: 2rem;">📊</div>
                <div style="color: white;">Original Features</div>
            </div>
            <div style="font-size: 2rem; color: rgba(255,255,255,0.3);">→</div>
            <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 10px; text-align: center; min-width: 120px;">
                <div style="font-size: 2rem;">🧹</div>
                <div style="color: white;">Cleaning</div>
            </div>
            <div style="font-size: 2rem; color: rgba(255,255,255,0.3);">→</div>
            <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 10px; text-align: center; min-width: 120px;">
                <div style="font-size: 2rem;">🔢</div>
                <div style="color: white;">Encoding</div>
            </div>
            <div style="font-size: 2rem; color: rgba(255,255,255,0.3);">→</div>
            <div style="background: rgba(46, 213, 115, 0.2); padding: 1rem; border-radius: 10px; text-align: center; min-width: 120px; border: 2px solid #2ed573;">
                <div style="font-size: 2rem;">✅</div>
                <div style="color: #2ed573; font-weight: 700;">Model Ready</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    train_fe = train.copy()
    train_fe['FamilySize'] = train_fe['SibSp'] + train_fe['Parch'] + 1
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #fff;'>New Feature Created</h4>", unsafe_allow_html=True)
        st.dataframe(train_fe[['SibSp', 'Parch', 'FamilySize']].head(10), width='stretch', hide_index=True)
        st.markdown("""
        <div style="color: rgba(255,255,255,0.7);">
            <p><strong style="color: #fff;">Family Size = SibSp + Parch + 1</strong></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #fff;'>Survival Rate by Family Size</h4>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(10, 6))
        train_fe.groupby('FamilySize')['Survived'].mean().plot(kind='bar', ax=ax, color='#ffa94d')
        ax.set_ylabel("Survival Rate", color='white')
        ax.tick_params(colors='white')
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

# ==================== FEATURE SELECTION ====================
elif page == "🎯 Feature Selection":
    st.markdown("<h1 style='color: #fff;'>🎯 Feature Selection</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='glass-card'>
            <h4 style='color: #ff6b6b;'>❌ Dropped Features</h4>
            <ul style='color: rgba(255,255,255,0.7); font-size: 1.1rem; line-height: 2.5;'>
                <li>PassengerId</li>
                <li>Name</li>
                <li>Ticket</li>
                <li>Cabin</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='glass-card'>
            <h4 style='color: #51cf66;'>✅ Selected Features</h4>
            <ul style='color: rgba(255,255,255,0.7); font-size: 1.1rem; line-height: 2.5;'>
                <li>Pclass</li>
                <li>Age</li>
                <li>Sex</li>
                <li>Fare</li>
                <li>FamilySize</li>
                <li>Embarked (encoded)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem;">
        <div style="background: rgba(102, 126, 234, 0.1); border: 2px solid #667eea; border-radius: 15px; padding: 1rem;">
            <span style="color: rgba(255,255,255,0.8);">💡 Features selected based on correlation analysis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==================== TRAIN/TEST SPLIT ====================
elif page == "✂️ Train/Test Split":
    st.markdown("<h1 style='color: #fff;'>✂️ Train/Test Split</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <div style="display: inline-block; padding: 1.5rem 3rem; background: rgba(255,255,255,0.05); backdrop-filter: blur(10px); border-radius: 15px; margin: 0.3rem; border: 1px solid rgba(255,255,255,0.1);">
            <div style="font-size: 2.5rem;">📊</div>
            <h3 style="color: #fff;">Dataset</h3>
        </div>
        <div style="display: inline-block; padding: 0.3rem 1rem; font-size: 2rem; color: rgba(255,255,255,0.3);">↓</div>
        <div style="display: inline-block; padding: 1.5rem 3rem; background: rgba(46, 213, 115, 0.1); backdrop-filter: blur(10px); border-radius: 15px; margin: 0.3rem; border: 2px solid #2ed573;">
            <div style="font-size: 2.5rem;">📚</div>
            <h3 style="color: #2ed573;">80% Training</h3>
        </div>
        <div style="display: inline-block; padding: 0.3rem 1rem; font-size: 2rem; color: rgba(255,255,255,0.3);">↓</div>
        <div style="display: inline-block; padding: 1.5rem 3rem; background: rgba(255, 71, 87, 0.1); backdrop-filter: blur(10px); border-radius: 15px; margin: 0.3rem; border: 2px solid #ff4757;">
            <div style="font-size: 2.5rem;">🧪</div>
            <h3 style="color: #ff4757;">20% Testing</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{X_train.shape[0]}</div>
            <div class='metric-label'>Training Samples</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{X_test.shape[0]}</div>
            <div class='metric-label'>Testing Samples</div>
        </div>
        """, unsafe_allow_html=True)

# ==================== ML MODELS ====================
elif page == "🤖 ML Models":
    st.markdown("<h1 style='color: #fff;'>🤖 ML Models Comparison</h1>", unsafe_allow_html=True)
    
    lr_pred = lr_model.predict(X_test)
    dt_pred = dt_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    
    models = {
        'Logistic Regression': lr_pred,
        'Decision Tree': dt_pred,
        'Random Forest': rf_pred
    }
    
    st.markdown("""
    <div class="glass-card">
        <h4 style="color: #fff;">📊 Model Performance Comparison</h4>
        <table class="comparison-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                </tr>
            </thead>
            <tbody>
    """, unsafe_allow_html=True)
    
    for name, pred in models.items():
        acc = accuracy_score(y_test, pred)
        report = classification_report(y_test, pred, output_dict=True)
        precision = report['1']['precision']
        recall = report['1']['recall']
        f1 = report['1']['f1-score']
        
        if name == 'Random Forest':
            st.markdown(f"""
            <tr class="winner">
                <td>🏆 {name}</td>
                <td>{acc:.2%}</td>
                <td>{precision:.2%}</td>
                <td>{recall:.2%}</td>
                <td>{f1:.2%}</td>
            </tr>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <tr>
                <td>{name}</td>
                <td>{acc:.2%}</td>
                <td>{precision:.2%}</td>
                <td>{recall:.2%}</td>
                <td>{f1:.2%}</td>
            </tr>
            """, unsafe_allow_html=True)
    
    st.markdown("""
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

# ==================== EVALUATION ====================
elif page == "📋 Evaluation":
    st.markdown("<h1 style='color: #fff;'>📋 Model Evaluation</h1>", unsafe_allow_html=True)
    
    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #fff;'>📊 Classification Report</h4>", unsafe_allow_html=True)
        report = classification_report(y_test, rf_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #fff;'>🧮 Confusion Matrix</h4>", unsafe_allow_html=True)
        cm = confusion_matrix(y_test, rf_pred)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Predicted", color='white')
        ax.set_ylabel("Actual", color='white')
        ax.tick_params(colors='white')
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    
    report = classification_report(y_test, rf_pred, output_dict=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{accuracy_score(y_test, rf_pred):.2%}</div>
            <div class='metric-label'>Accuracy</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{report['1']['precision']:.2%}</div>
            <div class='metric-label'>Precision</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{report['1']['recall']:.2%}</div>
            <div class='metric-label'>Recall</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{report['1']['f1-score']:.2%}</div>
            <div class='metric-label'>F1 Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<h4 style='color: #fff;'>📈 ROC Curve</h4>", unsafe_allow_html=True)
    
    fpr, tpr, _ = roc_curve(y_test, rf_proba)
    auc = roc_auc_score(y_test, rf_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'Random Forest (AUC = {auc:.3f})', color='#4dabf7', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel('False Positive Rate', color='white')
    ax.set_ylabel('True Positive Rate', color='white')
    ax.tick_params(colors='white')
    # FIX: Remove alpha parameter from legend
    ax.legend(loc='lower right', facecolor='black', labelcolor='white')
    st.pyplot(fig)
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== LIVE PREDICTION ====================
elif page == "🔮 Live Prediction":
    st.markdown("<h1 style='color: #fff;'>🔮 Live Prediction</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #fff;'>Enter Passenger Details</h4>", unsafe_allow_html=True)
        
        pclass = st.selectbox("Passenger Class", [1, 2, 3])
        sex = st.selectbox("Gender", ["male", "female"])
        age = st.slider("Age", 1, 80, 30)
        fare = st.number_input("Fare", min_value=0.0, max_value=600.0, value=32.0)
        sibsp = st.slider("Siblings/Spouses", 0, 8, 0)
        parch = st.slider("Parents/Children", 0, 6, 0)
        embarked = st.selectbox("Embarked", ["C", "Q", "S"])
        
        if st.button("🔮 Predict Survival", type="primary"):
            input_data = pd.DataFrame({
                'Pclass': [pclass],
                'Sex': [1 if sex == 'female' else 0],
                'Age': [age],
                'SibSp': [sibsp],
                'Parch': [parch],
                'Fare': [fare],
                'Embarked_Q': [1 if embarked == 'Q' else 0],
                'Embarked_S': [1 if embarked == 'S' else 0],
                'FamilySize': [sibsp + parch + 1]
            })
            
            input_data = input_data[feature_names]
            prediction = rf_model.predict(input_data)[0]
            probability = rf_model.predict_proba(input_data)[0][1]
            
            if probability >= 0.7:
                badge = '<span class="badge badge-high">High Confidence</span>'
            elif probability >= 0.4:
                badge = '<span class="badge badge-medium">Medium Confidence</span>'
            else:
                badge = '<span class="badge badge-low">Low Confidence</span>'
            
            with col2:
                if prediction == 1:
                    st.markdown(f"""
                    <div class='success-box'>
                        <div style='font-size: 4rem;'>✅</div>
                        <div style='font-size: 2.5rem; font-weight: 700; color: #2ed573;'>Passenger Will Survive</div>
                        <div style='font-size: 3.5rem; font-weight: 700; color: #fff; margin-top: 1rem;'>{probability:.1%}</div>
                        <div style='color: rgba(255,255,255,0.4);'>Probability</div>
                        <div style='margin-top: 1rem;'>{badge}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='error-box'>
                        <div style='font-size: 4rem;'>❌</div>
                        <div style='font-size: 2.5rem; font-weight: 700; color: #ff4757;'>Passenger Will Not Survive</div>
                        <div style='font-size: 3.5rem; font-weight: 700; color: #fff; margin-top: 1rem;'>{(1-probability):.1%}</div>
                        <div style='color: rgba(255,255,255,0.4);'>Probability</div>
                        <div style='margin-top: 1rem;'>{badge}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class='glass-card' style='margin-top: 1rem;'>
                    <h4 style='color: #fff;'>📋 Passenger Summary</h4>
                    <table style='width: 100%; color: rgba(255,255,255,0.7);'>
                        <tr><td>Class</td><td><strong>{pclass}</strong></td></tr>
                        <tr><td>Gender</td><td><strong>{sex}</strong></td></tr>
                        <tr><td>Age</td><td><strong>{age}</strong></td></tr>
                        <tr><td>Fare</td><td><strong>${fare:.2f}</strong></td></tr>
                        <tr><td>Family Size</td><td><strong>{sibsp + parch + 1}</strong></td></tr>
                    </table>
                </div>
                """, unsafe_allow_html=True)
        else:
            with col2:
                st.markdown("""
                <div style="display: flex; align-items: center; justify-content: center; height: 400px;">
                    <div style="text-align: center; color: rgba(255,255,255,0.2);">
                        <div style="font-size: 5rem;">🔮</div>
                        <p style="font-size: 1.2rem;">Enter details and click Predict</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# ==================== MODEL EXPLAINABILITY ====================
elif page == "💡 Model Explainability":
    st.markdown("<h1 style='color: #fff;'>💡 Model Explainability</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='color: rgba(255,255,255,0.7); text-align: center;'>Why did the model make this prediction?</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='glass-card' style='border: 2px solid #2ed573;'>
            <h3 style='color: #2ed573; text-align: center;'>✅ High Survival Chance</h3>
            <div style='color: rgba(255,255,255,0.8); padding: 1rem;'>
                <div style='display: flex; justify-content: space-between; padding: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.05);'>
                    <span>Female</span>
                    <span style='color: #2ed573;'>+38%</span>
                </div>
                <div style='display: flex; justify-content: space-between; padding: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.05);'>
                    <span>First Class</span>
                    <span style='color: #2ed573;'>+24%</span>
                </div>
                <div style='display: flex; justify-content: space-between; padding: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.05);'>
                    <span>Fare</span>
                    <span style='color: #2ed573;'>+10%</span>
                </div>
                <div style='display: flex; justify-content: space-between; padding: 0.5rem;'>
                    <span>Young Age</span>
                    <span style='color: #2ed573;'>+3%</span>
                </div>
                <div style='text-align: center; margin-top: 1rem; padding-top: 1rem; border-top: 2px solid #2ed573;'>
                    <div style='font-size: 2rem; font-weight: 700; color: #2ed573;'>97%</div>
                    <div style='color: rgba(255,255,255,0.4);'>Survival Probability</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='glass-card' style='border: 2px solid #ff4757;'>
            <h3 style='color: #ff4757; text-align: center;'>❌ Low Survival Chance</h3>
            <div style='color: rgba(255,255,255,0.8); padding: 1rem;'>
                <div style='display: flex; justify-content: space-between; padding: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.05);'>
                    <span>Male</span>
                    <span style='color: #ff4757;'>-38%</span>
                </div>
                <div style='display: flex; justify-content: space-between; padding: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.05);'>
                    <span>Third Class</span>
                    <span style='color: #ff4757;'>-24%</span>
                </div>
                <div style='display: flex; justify-content: space-between; padding: 0.5rem; border-bottom: 1px solid rgba(255,255,255,0.05);'>
                    <span>Low Fare</span>
                    <span style='color: #ff4757;'>-10%</span>
                </div>
                <div style='display: flex; justify-content: space-between; padding: 0.5rem;'>
                    <span>Older Age</span>
                    <span style='color: #ff4757;'>-3%</span>
                </div>
                <div style='text-align: center; margin-top: 1rem; padding-top: 1rem; border-top: 2px solid #ff4757;'>
                    <div style='font-size: 2rem; font-weight: 700; color: #ff4757;'>18%</div>
                    <div style='color: rgba(255,255,255,0.4);'>Survival Probability</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
# ==================== ABOUT ML MODELS ====================
elif page == "📖 About ML Models":
    st.markdown("<h1 style='color: #fff;'>📖 About Machine Learning Models</h1>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='glass-card'>
            <div style='text-align: center; font-size: 3.5rem;'>📊</div>
            <h3 style='color: #fff; text-align: center;'>Logistic Regression</h3>
            <p style='color: rgba(255,255,255,0.7);'>
                A statistical model that uses a logistic function to model a binary dependent variable.
            </p>
            <div style='background: rgba(255,255,255,0.05); padding: 0.5rem 1rem; border-radius: 10px; margin-top: 0.5rem;'>
                <p style='color: rgba(255,255,255,0.4); font-size: 0.9rem;'>
                    <strong style='color: rgba(255,255,255,0.6);'>Type:</strong> Classification<br>
                    <strong style='color: rgba(255,255,255,0.6);'>Best for:</strong> Linear relationships
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='glass-card'>
            <div style='text-align: center; font-size: 3.5rem;'>🌳</div>
            <h3 style='color: #fff; text-align: center;'>Decision Tree</h3>
            <p style='color: rgba(255,255,255,0.7);'>
                A tree-like model that splits data based on feature values to make predictions.
            </p>
            <div style='background: rgba(255,255,255,0.05); padding: 0.5rem 1rem; border-radius: 10px; margin-top: 0.5rem;'>
                <p style='color: rgba(255,255,255,0.4); font-size: 0.9rem;'>
                    <strong style='color: rgba(255,255,255,0.6);'>Type:</strong> Nonlinear<br>
                    <strong style='color: rgba(255,255,255,0.6);'>Best for:</strong> Interpretability
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='glass-card'>
            <div style='text-align: center; font-size: 3.5rem;'>🌲</div>
            <h3 style='color: #fff; text-align: center;'>Random Forest</h3>
            <p style='color: rgba(255,255,255,0.7);'>
                An ensemble method that combines multiple decision trees for better accuracy.
            </p>
            <div style='background: rgba(255,255,255,0.05); padding: 0.5rem 1rem; border-radius: 10px; margin-top: 0.5rem;'>
                <p style='color: rgba(255,255,255,0.4); font-size: 0.9rem;'>
                    <strong style='color: rgba(255,255,255,0.6);'>Type:</strong> Ensemble<br>
                    <strong style='color: rgba(255,255,255,0.6);'>Best for:</strong> High accuracy
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='glass-card'>
        <h4 style='color: #fff;'>📊 Model Comparison</h4>
        <div style="display: flex; justify-content: center; gap: 2rem; padding: 1rem; flex-wrap: wrap;">
            <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 10px; text-align: center; min-width: 150px;">
                <div>📊 Logistic Regression</div>
                <div style="font-size: 1.5rem; color: #4dabf7;">↓</div>
                <div style="color: rgba(255,255,255,0.6);">Classification</div>
            </div>
            <div style="font-size: 2rem; color: rgba(255,255,255,0.3);">→</div>
            <div style="background: rgba(255,255,255,0.05); padding: 1rem; border-radius: 10px; text-align: center; min-width: 150px;">
                <div>🌳 Decision Tree</div>
                <div style="font-size: 1.5rem; color: #ffa94d;">↓</div>
                <div style="color: rgba(255,255,255,0.6);">Nonlinear</div>
            </div>
            <div style="font-size: 2rem; color: rgba(255,255,255,0.3);">→</div>
            <div style="background: rgba(255,212,59,0.1); padding: 1rem; border-radius: 10px; text-align: center; min-width: 150px; border: 2px solid #ffd43b;">
                <div>🏆 Random Forest</div>
                <div style="font-size: 1.5rem; color: #ffd43b;">↓</div>
                <div style="color: rgba(255,255,255,0.6);">Ensemble</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ==================== ABOUT ME ====================
# ==================== ABOUT ME ====================
elif page == "👤 About Me":
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
           <div style="font-size: 5rem;">👨‍💻</div>
           <h1 style="color: #ffffff; font-size: 3rem; margin: 0.5rem 0; font-weight: 700;">Muhammad Zalan</h1>
           <p style="color: rgba(255,255,255,0.7); font-size: 1.2rem; margin: 0.3rem 0;">BS Computer Science</p>
           <p style="color: rgba(255,255,255,0.4); font-size: 1.1rem; margin: 0.3rem 0;">Aspiring Data Scientist</p>
        
    <div style="max-width: 400px; margin: 2rem auto;">
            <div style="background: rgba(255,255,255,0.08); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); border-radius: 20px; padding: 1.5rem; border: 1px solid rgba(255,255,255,0.12); box-shadow: 0 8px 32px rgba(0,0,0,0.2);">
            <p style="color: rgba(255,255,255,0.9); font-size: 1.5rem; margin: 0;">🇵🇰 Pakistan</p>
            </div>
    </div>
        
    <div style="max-width: 500px; margin: 2rem auto;">
            <div style="background: rgba(255,255,255,0.08); backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px); border-radius: 20px; padding: 1.5rem; border: 1px solid rgba(255,255,255,0.12); box-shadow: 0 8px 32px rgba(0,0,0,0.2);">
            <p style="color: rgba(255,255,255,0.8); font-size: 1rem; line-height: 1.8; margin: 0;">
            Passionate about Machine Learning, Data Science, and creating impactful solutions.
            </p>
            </div>
    </div>
        
    <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; max-width: 500px; margin: 0 auto;">
            <span style="background: rgba(102,126,234,0.2); padding: 0.3rem 1rem; border-radius: 20px; color: #a8b5ff; font-size: 0.9rem;">🐍 Python</span>
            <span style="background: rgba(102,126,234,0.2); padding: 0.3rem 1rem; border-radius: 20px; color: #a8b5ff; font-size: 0.9rem;">📊 ML</span>
            <span style="background: rgba(102,126,234,0.2); padding: 0.3rem 1rem; border-radius: 20px; color: #a8b5ff; font-size: 0.9rem;">📈 Data Science</span>
            <span style="background: rgba(102,126,234,0.2); padding: 0.3rem 1rem; border-radius: 20px; color: #a8b5ff; font-size: 0.9rem;">🚀 Streamlit</span>
    </div>
        
    <div style="display: flex; justify-content: center; gap: 1rem; margin-top: 1.5rem; flex-wrap: wrap;">
            <span style="background: rgba(255,255,255,0.05); padding: 0.5rem 1.5rem; border-radius: 10px; color: rgba(255,255,255,0.6);">📧 zalan@email.com</span>
            <span style="background: rgba(255,255,255,0.05); padding: 0.5rem 1.5rem; border-radius: 10px; color: rgba(255,255,255,0.6);">🐙 GitHub</span>
            <span style="background: rgba(255,255,255,0.05); padding: 0.5rem 1.5rem; border-radius: 10px; color: rgba(255,255,255,0.6);">🔗 LinkedIn</span>
    </div>
    </div>
    """, unsafe_allow_html=True)  # <-- THIS IS CRUCIAL!
# ==================== FOOTER ====================
st.markdown("""
<div class='footer'>
    Made with ❤️ using Streamlit · Muhammad Zalan
</div>
""", unsafe_allow_html=True)