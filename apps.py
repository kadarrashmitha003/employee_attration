# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
import hashlib
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="TechNova Employee Attrition Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with enhanced styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem; 
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #1f77b4, #2ca02c);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .report-title {
        font-size: 2rem; 
        color: #1f77b4; 
        text-align: center;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .feature-box {
        border-radius: 8px; 
        padding: 15px; 
        background-color: #f0f8ff; 
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #1f77b4;
    }
    .prediction-safe {
        color: #2ca02c; 
        font-weight: bold;
        font-size: 1.2rem;
        padding: 10px;
        background-color: #f0fff0;
        border-radius: 5px;
        text-align: center;
    }
    .prediction-medium {
        color: #ff7f0e; 
        font-weight: bold;
        font-size: 1.2rem;
        padding: 10px;
        background-color: #fff5e6;
        border-radius: 5px;
        text-align: center;
    }
    .prediction-risk {
        color: #d62728; 
        font-weight: bold;
        font-size: 1.2rem;
        padding: 10px;
        background-color: #fff0f0;
        border-radius: 5px;
        text-align: center;
    }
    .stProgress > div > div > div > div {
        background-color: #1f77b4;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1f77b4, #2ca02c);
    }
    .sidebar .sidebar-content .block-container {
        color: white;
    }
    .nav-button {
        background-color: #1f77b4;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .nav-button:hover {
        background-color: #2ca02c;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .nav-button.active {
        background-color: #2ca02c;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        border-left: 4px solid #1f77b4;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #6c757d;
    }
    .login-container {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        background: linear-gradient(135deg, #1f77b4, #2ca02c);
    }
    .login-form {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        width: 100%;
        max-width: 400px;
    }
    .login-title {
        text-align: center;
        color: #d4edda ;
        margin-bottom: 1.5rem;
        font-size: 1.8rem;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        padding: 8px;
        border-radius: 4px;
        text-align: center;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 8px;
        border-radius: 4px;
        text-align: center;
        font-weight: bold;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 8px;
        border-radius: 4px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# User authentication
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password, hashed_text):
    return make_hashes(password) == hashed_text

# Database simulation (in production, use a real database)
def create_user_table():
    if 'users' not in st.session_state:
        st.session_state.users = {
            'admin': make_hashes('admin123'),
            'hr_manager': make_hashes('hr123'),
            'team_lead': make_hashes('lead123')
        }

def login_page():
    st.markdown("""
    <div class="login-container">
        <div class="login-form">
            <h2 class="login-title">🔐 TechNova Login</h2>
    """, unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type='password', placeholder="Enter your password")
        submitted = st.form_submit_button("🚀 Login", use_container_width=True)
        
        if submitted:
            create_user_table()
            if username in st.session_state.users:
                if check_hashes(password, st.session_state.users[username]):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.success(f"✅ Logged in as {username}")
                    st.rerun()
                else:
                    st.error("❌ Incorrect password")
            else:
                st.error("❌ Username not found")
    
    st.markdown("</div></div>", unsafe_allow_html=True)

def logout_user():
    st.session_state.logged_in = False
    st.session_state.username = None

# Load data from CSV file
@st.cache_data
def load_data():
    try:
        # Load the dataset from the provided CSV file
        df = pd.read_csv(r"C:\Users\KADARUS\OneDrive\Desktop\MLproject\MLproject\rawdata\technova_attrition_dataset.csv")
        
        # Map the target variable to 0 and 1 if it's categorical
        if 'Attrition' in df.columns:
            df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})
        
        # Rename columns to match the expected format in the app
        column_mapping = {
            'Age': 'age',
            'JobSatisfaction': 'job_satisfaction',
            'MonthlyIncome': 'salary',
            'YearsAtCompany': 'tenure',
            'EnvironmentSatisfaction': 'work_env_satisfaction',
            'OverTime': 'overtime',
            'MaritalStatus': 'marital_status',
            'Education': 'education',
            'Department': 'department',
            'YearsSinceLastPromotion': 'years_since_last_promotion',
            'TrainingTimesLastYear': 'training_hours',
            'WorkLifeBalance': 'work_life_balance',
            'Attrition': 'attrition'
        }
        
        df = df.rename(columns=column_mapping)
        
        # Ensure all required columns are present
        required_columns = ['age', 'job_satisfaction', 'salary', 'tenure', 'work_env_satisfaction', 
                          'overtime', 'marital_status', 'education', 'department', 
                          'years_since_last_promotion', 'training_hours', 'work_life_balance', 'attrition']
        
        # Add missing columns with default values if necessary
        for col in required_columns:
            if col not in df.columns:
                if col == 'promotion_last_5years':
                    df[col] = (df['years_since_last_promotion'] <= 5).astype(int)
                else:
                    df[col] = 0  # Default value for missing columns
        
        return df
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.info("Using sample data instead.")
        return load_sample_data()

# Sample data generation (fallback if CSV loading fails)
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(22, 65, n_samples),
        'job_satisfaction': np.random.randint(1, 5, n_samples),
        'salary': np.round(np.random.normal(75000, 25000, n_samples), 2),
        'tenure': np.random.randint(0, 15, n_samples),
        'work_env_satisfaction': np.random.randint(1, 5, n_samples),
        'overtime': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'education': np.random.choice(['Bachelor', 'Master', 'PhD', 'Other'], n_samples),
        'department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing', 'R&D'], n_samples),
        'promotion_last_5years': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
        'years_since_last_promotion': np.random.randint(0, 10, n_samples),
        'training_hours': np.random.randint(0, 100, n_samples),
        'work_life_balance': np.random.randint(1, 5, n_samples),
    }
    
    # Create a target variable based on some rules
    df = pd.DataFrame(data)
    
    # Simulate attrition probability
    attrition_prob = (
        0.1 * (df['job_satisfaction'] < 2) +
        0.15 * (df['work_env_satisfaction'] < 2) +
        0.2 * (df['work_life_balance'] < 2) +
        0.1 * (df['overtime'] == 'Yes') +
        0.05 * (df['years_since_last_promotion'] > 5) -
        0.1 * (df['salary'] > 90000) -
        0.05 * (df['tenure'] > 5)
    )
    
    df['attrition'] = np.random.binomial(1, attrition_prob.clip(0, 0.9))
    return df

# Train model
@st.cache_resource
def train_model(df):
    # Prepare features
    X = df.drop('attrition', axis=1)
    X = pd.get_dummies(X)
    y = df['attrition']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return model, X.columns, accuracy

# Navigation component
def render_navigation(current_page):
    cols = st.columns(5)
    pages = [
        ("📊 Dashboard", "Dashboard"),
        ("👤 Single Prediction", "Single Prediction"),
        ("📁 Batch Prediction", "Batch Prediction"),
        ("🔍 Data Insights", "Data Insights"),
        ("💡 Retention Strategies", "Retention Strategies")
    ]
    
    for i, (icon_name, page_name) in enumerate(pages):
        with cols[i]:
            if st.button(
                icon_name, 
                use_container_width=True, 
                type="primary" if current_page == page_name else "secondary"
            ):
                st.session_state.current_page = page_name
                st.rerun()

# Risk categorization function
def categorize_risk(probability):
    if probability < 0.3:
        return "Low", "risk-low"
    elif probability < 0.7:
        return "Medium", "risk-medium"
    else:
        return "High", "risk-high"

# Main app
def main():
    # Check if user is logged in
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    
    if not st.session_state.logged_in:
        login_page()
        return
    
    st.markdown('<h1 class="main-header">TechNova Employee Attrition Predictor</h1>', unsafe_allow_html=True)
    
    # Initialize current page if not set
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    # Sidebar with user info and logout
    st.sidebar.markdown(f"### 👋 Welcome, {st.session_state.username}!")
    st.sidebar.markdown("---")
    st.sidebar.button("🚪 Logout", on_click=logout_user, use_container_width=True)
    
    # Load data
    df = load_data()
    
    # Train model
    model, feature_columns, accuracy = train_model(df)
    
    # Render navigation
    render_navigation(st.session_state.current_page)
    
    # Page routing
    if st.session_state.current_page == "Dashboard":
        show_dashboard(df, accuracy)
    elif st.session_state.current_page == "Single Prediction":
        predict_attrition(model, feature_columns, df)
    elif st.session_state.current_page == "Batch Prediction":
        batch_prediction(model, feature_columns, df)
    elif st.session_state.current_page == "Data Insights":
        show_employee_analysis(df)
    elif st.session_state.current_page == "Retention Strategies":
        show_retention_strategies()

def show_dashboard(df, accuracy):
    st.markdown('<h2 class="report-title">📊 Employee Attrition Dashboard</h2>', unsafe_allow_html=True)
    
    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    
    attrition_rate = df['attrition'].mean() * 100
    avg_tenure = df['tenure'].mean()
    avg_salary = df['salary'].mean()
    avg_job_satisfaction = df['job_satisfaction'].mean()
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{attrition_rate:.2f}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Attrition Rate</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{avg_tenure:.1f} years</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Average Tenure</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">${avg_salary:,.2f}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Average Salary</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{avg_job_satisfaction:.1f}/5</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Avg Job Satisfaction</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Attrition by department
        dept_attrition = df.groupby('department')['attrition'].mean().reset_index()
        fig = px.bar(dept_attrition, x='department', y='attrition', 
                    title='📈 Attrition Rate by Department',
                    labels={'attrition': 'Attrition Rate', 'department': 'Department'},
                    color='attrition', color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Job satisfaction vs attrition
        sat_attrition = df.groupby('job_satisfaction')['attrition'].mean().reset_index()
        fig = px.line(sat_attrition, x='job_satisfaction', y='attrition', 
                     title='📉 Attrition Rate by Job Satisfaction Level',
                     labels={'attrition': 'Attrition Rate', 'job_satisfaction': 'Job Satisfaction'},
                     markers=True)
        fig.update_traces(line=dict(color='#1f77b4', width=3))
        st.plotly_chart(fig, use_container_width=True)
    
    # Additional charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Salary distribution by department
        fig = px.box(df, x='department', y='salary', 
                    title='💰 Salary Distribution by Department',
                    color='department', color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Attrition by tenure
        tenure_attrition = df.groupby('tenure')['attrition'].mean().reset_index()
        fig = px.area(tenure_attrition, x='tenure', y='attrition',
                     title='📅 Attrition Rate by Tenure',
                     labels={'attrition': 'Attrition Rate', 'tenure': 'Years of Tenure'})
        fig.update_traces(line=dict(color='#ff7f0e', width=3))
        st.plotly_chart(fig, use_container_width=True)
    
    # Model accuracy
    st.markdown("---")
    st.subheader("🤖 Prediction Model Performance")
    st.write(f"Current model accuracy: **{accuracy:.2%}**")
    
    # Feature importance
    st.subheader("🎯 Top Factors Influencing Attrition")
    factors = [
        {"factor": "Low Job Satisfaction", "impact": "High", "icon": "😔"},
        {"factor": "Poor Work Environment", "impact": "High", "icon": "🏢"},
        {"factor": "Work-Life Balance Issues", "impact": "High", "icon": "⚖️"},
        {"factor": "Lack of Career Growth", "impact": "Medium", "icon": "📈"},
        {"factor": "Low Compensation", "impact": "Medium", "icon": "💰"},
        {"factor": "Lack of Recognition", "impact": "Medium", "icon": "👏"},
    ]
    
    for f in factors:
        with st.expander(f"{f['icon']} {f['factor']} ({f['impact']} Impact)"):
            st.write("**Suggested interventions:**")
            if f['factor'] == "Low Job Satisfaction":
                st.write("- 📝 Regular feedback sessions\n- 🎯 Clear career paths\n- ✨ Meaningful work assignments")
            elif f['factor'] == "Poor Work Environment":
                st.write("- 🤝 Team building activities\n- 💬 Improved communication\n- 🕊️ Conflict resolution programs")
            elif f['factor'] == "Work-Life Balance Issues":
                st.write("- 🏠 Flexible work arrangements\n- 🌐 Remote work options\n- 💪 Wellness programs")
            elif f['factor'] == "Lack of Career Growth":
                st.write("- 📚 Training programs\n- 🎓 Mentorship opportunities\n- 🔄 Job rotation")
            elif f['factor'] == "Low Compensation":
                st.write("- 💰 Competitive salary reviews\n- 🎯 Performance bonuses\n- 📈 Equity programs")
            elif f['factor'] == "Lack of Recognition":
                st.write("- 🏆 Employee recognition programs\n- 📢 Success sharing\n- 🎁 Reward systems")

def show_employee_analysis(df):
    st.markdown('<h2 class="report-title">🔍 Employee Data Insights</h2>', unsafe_allow_html=True)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        departments = st.multiselect("🏢 Select Departments", options=df['department'].unique(), 
                                    default=df['department'].unique())
    with col2:
        min_salary, max_salary = st.slider("💰 Salary Range", 
                                          min_value=int(df['salary'].min()), 
                                          max_value=int(df['salary'].max()),
                                          value=(int(df['salary'].min()), int(df['salary'].max())))
    with col3:
        satisfaction_filter = st.slider("😊 Minimum Job Satisfaction", 1, 5, 1)
    
    # Additional filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        education_filter = st.multiselect("🎓 Education Level", options=df['education'].unique(), 
                                         default=df['education'].unique())
    with col2:
        tenure_filter = st.slider("📅 Tenure Range (years)", 
                                 min_value=int(df['tenure'].min()), 
                                 max_value=int(df['tenure'].max()),
                                 value=(int(df['tenure'].min()), int(df['tenure'].max())))
    with col3:
        attrition_filter = st.selectbox("🔍 Attrition Status", 
                                       options=['All', 'Left', 'Stayed'], 
                                       index=0)
    
    # Apply filters
    filtered_df = df[
        (df['department'].isin(departments)) &
        (df['salary'] >= min_salary) &
        (df['salary'] <= max_salary) &
        (df['job_satisfaction'] >= satisfaction_filter) &
        (df['education'].isin(education_filter)) &
        (df['tenure'] >= tenure_filter[0]) &
        (df['tenure'] <= tenure_filter[1])
    ]
    
    if attrition_filter == 'Left':
        filtered_df = filtered_df[filtered_df['attrition'] == 1]
    elif attrition_filter == 'Stayed':
        filtered_df = filtered_df[filtered_df['attrition'] == 0]
    
    # Show summary stats
    st.subheader("📋 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Employees", len(filtered_df))
    with col2:
        attrition_rate = filtered_df['attrition'].mean() * 100
        st.metric("Attrition Rate", f"{attrition_rate:.2f}%")
    with col3:
        avg_salary = filtered_df['salary'].mean()
        st.metric("Average Salary", f"${avg_salary:,.2f}")
    with col4:
        avg_satisfaction = filtered_df['job_satisfaction'].mean()
        st.metric("Avg Satisfaction", f"{avg_satisfaction:.1f}/5")
    
    # Show filtered data
    st.subheader("📊 Filtered Employee Data")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button("📥 Download Filtered Data", data=csv, file_name="employee_data.csv", mime="text/csv")
    
    # Correlation analysis
    st.subheader("📈 Factor Correlation Analysis")
    
    # Select features for correlation
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.multiselect("Select features for correlation analysis", 
                                      options=numeric_cols,
                                      default=['job_satisfaction', 'work_env_satisfaction', 'work_life_balance', 'tenure', 'salary'])
    
    if selected_features:
        corr_matrix = filtered_df[selected_features + ['attrition']].corr()
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto", 
                       title="Correlation Matrix", color_continuous_scale='RdBu_r')
        st.plotly_chart(fig, use_container_width=True)
    
    # Advanced visualizations
    st.subheader("🔍 Advanced Visualizations")
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter plot
        x_axis = st.selectbox("X-Axis", options=numeric_cols, index=numeric_cols.index('salary'))
        y_axis = st.selectbox("Y-Axis", options=numeric_cols, index=numeric_cols.index('job_satisfaction'))
        color_by = st.selectbox("Color By", options=['attrition', 'department', 'education'])
        
        fig = px.scatter(filtered_df, x=x_axis, y=y_axis, color=color_by,
                        title=f"{y_axis} vs {x_axis} by {color_by}",
                        hover_data=['age', 'tenure', 'department'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Distribution plot
        dist_feature = st.selectbox("Distribution Feature", options=numeric_cols, index=numeric_cols.index('age'))
        color_by = st.selectbox("Split By", options=['None', 'attrition', 'department', 'education'])
        
        if color_by == 'None':
            fig = px.histogram(filtered_df, x=dist_feature, title=f"Distribution of {dist_feature}")
        else:
            fig = px.histogram(filtered_df, x=dist_feature, color=color_by,
                              title=f"Distribution of {dist_feature} by {color_by}",
                              barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)

def predict_attrition(model, feature_columns, df):
    st.markdown('<h2 class="report-title">👤 Single Employee Prediction</h2>', unsafe_allow_html=True)
    
    # Input form
    with st.form("employee_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Personal Details")
            age = st.slider("🎂 Age", 20, 65, 30)
            department = st.selectbox("🏢 Department", options=df['department'].unique())
            education = st.selectbox("🎓 Education Level", options=df['education'].unique())
            marital_status = st.selectbox("💑 Marital Status", options=df['marital_status'].unique())
            tenure = st.slider("📅 Tenure (years)", 0, 15, 3)
        
        with col2:
            st.subheader("💼 Work Details")
            salary = st.number_input("💰 Salary ($)", min_value=30000, max_value=200000, value=75000, step=5000)
            job_satisfaction = st.slider("😊 Job Satisfaction (1-5)", 1, 5, 3)
            work_env_satisfaction = st.slider("🏢 Work Environment Satisfaction (1-5)", 1, 5, 3)
            work_life_balance = st.slider("⚖️ Work-Life Balance (1-5)", 1, 5, 3)
            overtime = st.selectbox("⏰ Works Overtime", options=['Yes', 'No'])
            training_hours = st.slider("📚 Training Hours (last year)", 0, 100, 20)
            promotion = st.selectbox("📈 Promoted in last 5 years?", options=['Yes', 'No'])
        
        submitted = st.form_submit_button("🎯 Predict Attrition Risk", use_container_width=True)
    
    if submitted:
        # Prepare input data
        input_data = {
            'age': age,
            'job_satisfaction': job_satisfaction,
            'salary': salary,
            'tenure': tenure,
            'work_env_satisfaction': work_env_satisfaction,
            'overtime': overtime,
            'marital_status': marital_status,
            'education': education,
            'department': department,
            'promotion_last_5years': 1 if promotion == 'Yes' else 0,
            'years_since_last_promotion': 0 if promotion == 'Yes' else np.random.randint(1, 6),
            'training_hours': training_hours,
            'work_life_balance': work_life_balance,
        }
        
        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df)
        
        # Ensure all training columns are present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        input_df = input_df[feature_columns]
        
        # Make prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)
        
        # Display results
        st.subheader("📊 Prediction Results")
        
        # Create a visual gauge for attrition risk
        risk_percentage = prediction_proba[0][1] * 100
        
        # Categorize risk
        risk_level, risk_class = categorize_risk(prediction_proba[0][1])
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = risk_percentage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Attrition Risk Score", 'font': {'size': 24}},
            delta = {'reference': 50, 'increasing': {'color': "RebeccaPurple"}},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkblue"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': 'lightgreen'},
                    {'range': [30, 70], 'color': 'yellow'},
                    {'range': [70, 100], 'color': 'pink'}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90}}))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Display risk category
        st.markdown(f'<div class="{risk_class}">Risk Level: {risk_level} ({prediction_proba[0][1]:.2%})</div>', unsafe_allow_html=True)
        
        if risk_level == "High":
            # Show risk factors
            st.write("**🔍 Key Risk Factors:**")
            risk_factors = []
            if job_satisfaction <= 2:
                risk_factors.append("- 😔 Low job satisfaction")
            if work_env_satisfaction <= 2:
                risk_factors.append("- 🏢 Poor work environment satisfaction")
            if work_life_balance <= 2:
                risk_factors.append("- ⚖️ Poor work-life balance")
            if overtime == 'Yes':
                risk_factors.append("- ⏰ Frequently works overtime")
            if promotion == 'No':
                risk_factors.append("- 📈 No recent promotion")
            if salary < df[df['department'] == department]['salary'].mean():
                risk_factors.append("- 💰 Below average salary for department")
            
            for factor in risk_factors:
                st.write(factor)
            
            # Recommendations
            st.write("**💡 Recommended Interventions:**")
            st.write("- 📅 Schedule one-on-one meeting to discuss concerns")
            st.write("- 🎯 Consider career development opportunities")
            st.write("- ⚖️ Evaluate workload and work-life balance")
            if salary < df[df['department'] == department]['salary'].mean():
                st.write("- 💰 Review compensation package")
            if training_hours < 30:
                st.write("- 📚 Increase training and development opportunities")
            
        elif risk_level == "Medium":
            st.write("⚠️ This employee shows some risk factors for attrition. Consider proactive engagement strategies.")
            
            # Strengths and weaknesses
            st.write("**🌟 Strengths:**")
            strengths = []
            if job_satisfaction >= 4:
                strengths.append("- 😊 High job satisfaction")
            if work_env_satisfaction >= 4:
                strengths.append("- 🏢 Positive work environment")
            if work_life_balance >= 4:
                strengths.append("- ⚖️ Good work-life balance")
            
            if not strengths:
                strengths.append("- No significant strengths identified")
                
            for strength in strengths:
                st.write(strength)
                
            st.write("**📉 Areas for Improvement:**")
            improvements = []
            if job_satisfaction <= 3:
                improvements.append("- 😔 Moderate job satisfaction")
            if work_env_satisfaction <= 3:
                improvements.append("- 🏢 Moderate work environment satisfaction")
            if work_life_balance <= 3:
                improvements.append("- ⚖️ Moderate work-life balance")
            if overtime == 'Yes':
                improvements.append("- ⏰ Occasionally works overtime")
                
            for improvement in improvements:
                st.write(improvement)
                
        else:
            st.write("🎉 This employee shows low risk factors for attrition. Continue with current engagement strategies.")
            
            # Strengths
            st.write("**🌟 Strengths:**")
            strengths = []
            if job_satisfaction >= 4:
                strengths.append("- 😊 High job satisfaction")
            if work_env_satisfaction >= 4:
                strengths.append("- 🏢 Positive work environment")
            if work_life_balance >= 4:
                strengths.append("- ⚖️ Good work-life balance")
            if promotion == 'Yes':
                strengths.append("- 📈 Recent promotion")
            if salary >= df[df['department'] == department]['salary'].mean():
                strengths.append("- 💰 Competitive salary")
            
            for strength in strengths:
                st.write(strength)

def batch_prediction(model, feature_columns, df):
    st.markdown('<h2 class="report-title">📁 Batch Employee Prediction</h2>', unsafe_allow_html=True)
    
    st.info("📝 Upload a CSV file containing employee data to get predictions for multiple employees at once.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            batch_df = pd.read_csv(uploaded_file)
            
            st.success("✅ File successfully uploaded!")
            st.write("**Preview of uploaded data:**")
            st.dataframe(batch_df.head())
            
            # Check if required columns are present
            required_columns = ['age', 'job_satisfaction', 'salary', 'tenure', 'work_env_satisfaction', 
                              'overtime', 'marital_status', 'education', 'department', 'work_life_balance']
            
            missing_columns = [col for col in required_columns if col not in batch_df.columns]
            
            if missing_columns:
                st.error(f"❌ Missing required columns: {', '.join(missing_columns)}")
                st.info("ℹ️ Please make sure your CSV file contains all required columns.")
            else:
                if st.button("🚀 Generate Predictions", use_container_width=True):
                    with st.spinner("🔮 Generating predictions..."):
                        # Prepare the data
                        prediction_df = batch_df.copy()
                        prediction_df = pd.get_dummies(prediction_df)
                        
                        # Ensure all training columns are present
                        for col in feature_columns:
                            if col not in prediction_df.columns:
                                prediction_df[col] = 0
                        
                        prediction_df = prediction_df[feature_columns]
                        
                        # Make predictions
                        predictions = model.predict(prediction_df)
                        prediction_probas = model.predict_proba(prediction_df)
                        
                        # Add predictions to the original dataframe
                        batch_df['Attrition_Risk_Probability'] = prediction_probas[:, 1]
                        batch_df['Attrition_Prediction'] = predictions
                        
                        # Add risk categorization
                        risk_levels = []
                        for prob in prediction_probas[:, 1]:
                            risk_level, _ = categorize_risk(prob)
                            risk_levels.append(risk_level)
                        
                        batch_df['Risk_Category'] = risk_levels
                        
                        # Display results
                        st.success("✅ Predictions generated successfully!")
                        
                        # Show summary
                        st.subheader("📊 Prediction Summary")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            low_risk = (batch_df['Risk_Category'] == 'Low').sum()
                            st.metric("Low Risk Employees", low_risk)
                        
                        with col2:
                            medium_risk = (batch_df['Risk_Category'] == 'Medium').sum()
                            st.metric("Medium Risk Employees", medium_risk)
                        
                        with col3:
                            high_risk = (batch_df['Risk_Category'] == 'High').sum()
                            st.metric("High Risk Employees", high_risk)
                        
                        # Show detailed results
                        st.subheader("📋 Detailed Predictions")
                        st.dataframe(batch_df)
                        
                        # Download results
                        csv = batch_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name="employee_attrition_predictions.csv",
                            mime="text/csv"
                        )
                        
                        # Visualizations
                        st.subheader("📈 Risk Distribution")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            risk_counts = batch_df['Risk_Category'].value_counts()
                            fig = px.pie(
                                values=risk_counts.values,
                                names=risk_counts.index,
                                title="Risk Category Distribution",
                                color=risk_counts.index,
                                color_discrete_map={'Low': '#2ca02c', 'Medium': '#ff7f0e', 'High': '#d62728'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = px.histogram(
                                batch_df, 
                                x='Attrition_Risk_Probability', 
                                nbins=20,
                                title="Distribution of Attrition Risk Scores",
                                labels={'Attrition_Risk_Probability': 'Attrition Risk Probability'}
                            )
                            st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
    else:
        # Show sample format
        st.subheader("📋 Expected CSV Format")
        st.write("Your CSV file should include the following columns with appropriate data:")
        
        sample_data = {
            'age': [32, 45, 28],
            'job_satisfaction': [4, 2, 3],
            'salary': [75000, 65000, 82000],
            'tenure': [3, 8, 2],
            'work_env_satisfaction': [4, 3, 2],
            'overtime': ['No', 'Yes', 'No'],
            'marital_status': ['Married', 'Single', 'Married'],
            'education': ['Bachelor', 'Master', 'Bachelor'],
            'department': ['IT', 'HR', 'Finance'],
            'work_life_balance': [4, 2, 3]
        }
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df)
        
        # Provide template download
        csv = sample_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Template",
            data=csv,
            file_name="employee_data_template.csv",
            mime="text/csv"
        )

def show_retention_strategies():
    st.markdown('<h2 class="report-title">💡 Employee Retention Strategies</h2>', unsafe_allow_html=True)
    
    st.subheader("🚀 Proactive Retention Initiatives")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Career Development")
        with st.expander("View Programs"):
            st.write("""
            - **👥 Mentorship Program**: Pair employees with senior leaders
            - **📚 Skill Development**: Budget for courses and certifications
            - **🎯 Career Pathing**: Clear progression frameworks for all roles
            - **🔄 Internal Mobility**: Priority consideration for open positions
            """)
        
        st.markdown("### 💰 Compensation & Benefits")
        with st.expander("View Initiatives"):
            st.write("""
            - **💵 Competitive Salary Reviews**: Quarterly market analysis
            - **🏆 Performance Bonuses**: Tie rewards to measurable outcomes
            - **📈 Equity Programs**: Stock options for key contributors
            - **🎁 Benefits Enhancement**: Flexible spending accounts
            """)
    
    with col2:
        st.markdown("### 🌟 Recognition & Culture")
        with st.expander("View Programs"):
            st.write("""
            - **👏 Employee Recognition**: Peer-to-peer recognition platform
            - **🤝 Culture Committees**: Employee-led culture initiatives
            - **🎉 Team Building**: Quarterly offsites and events
            - **💬 Feedback Culture**: Regular pulse surveys and action planning
            """)
        
        st.markdown("### ⚖️ Work-Life Balance")
        with st.expander("View Initiatives"):
            st.write("""
            - **🏠 Flexible Work Arrangements**: Remote and hybrid options
            - **🏖️ Unlimited PTO**: Trust-based time off policy
            - **💪 Wellness Programs**: Mental and physical health support
            - **👨‍👩‍👧‍👦 Family Support**: Parental leave and childcare assistance
            """)
    
    st.markdown("---")
    st.subheader("📊 Retention Program Effectiveness")
    
    # Simulated program effectiveness data
    programs = {
        'Program': ['Mentorship', 'Flex Work', 'Training Budget', 'Salary Increase', 'Recognition'],
        'Retention Improvement (%)': [15, 12, 10, 18, 8],
        'Cost': ['Low', 'Low', 'Medium', 'High', 'Low'],
        'Implementation Time (months)': [2, 1, 3, 6, 1]
    }
    
    program_df = pd.DataFrame(programs)
    
    # Bar chart
    fig = px.bar(program_df, x='Program', y='Retention Improvement (%)', color='Cost',
                 title='Estimated Retention Improvement by Program',
                 color_discrete_sequence=px.colors.qualitative.Set2)
    st.plotly_chart(fig, use_container_width=True)
    
    # Before-after comparison
    st.subheader("📈 Before & After Implementation Analysis")
    
    # Simulated before-after data
    strategies = ['Career Development', 'Flexible Work', 'Enhanced Benefits', 'Recognition Programs']
    before = [65, 68, 62, 60]  # Simulated retention rates before
    after = [80, 80, 75, 68]   # Simulated retention rates after
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Before Implementation',
        x=strategies,
        y=before,
        marker_color='#1f77b4'
    ))
    fig.add_trace(go.Bar(
        name='After Implementation',
        x=strategies,
        y=after,
        marker_color='#2ca02c'
    ))
    
    fig.update_layout(
        title='Retention Rates Before and After Strategy Implementation',
        xaxis_title='Retention Strategy',
        yaxis_title='Retention Rate (%)',
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # ROI Analysis
    st.subheader("📊 Return on Investment Analysis")
    
    # Simulated ROI data
    roi_data = {
        'Strategy': ['Mentorship Program', 'Flexible Work', 'Training Budget', 'Salary Increase'],
        'Cost ($)': [5000, 2000, 15000, 50000],
        'Benefits ($)': [25000, 18000, 30000, 75000],
        'ROI (%)': [400, 800, 100, 50]
    }
    
    roi_df = pd.DataFrame(roi_data)
    
    fig = px.scatter(roi_df, x='Cost ($)', y='Benefits ($)', size='ROI (%)', color='Strategy',
                    title='ROI of Different Retention Strategies',
                    hover_name='Strategy', size_max=60,
                    color_discrete_sequence=px.colors.qualitative.Set3)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Implementation timeline
    st.subheader("📅 Recommended Implementation Timeline")
    
    timeline_data = {
        'Task': ['Needs Assessment', 'Program Design', 'Pilot Implementation', 'Full Rollout', 'Evaluation'],
        'Start': ['2023-01-01', '2023-02-01', '2023-04-01', '2023-07-01', '2023-10-01'],
        'Finish': ['2023-01-31', '2023-03-31', '2023-06-30', '2023-09-30', '2023-12-31'],
        'Completion': [100, 100, 75, 25, 0]
    }
    
    timeline_df = pd.DataFrame(timeline_data)
    
    fig = px.timeline(timeline_df, x_start="Start", x_end="Finish", y="Task", 
                     title="Retention Program Implementation Timeline",
                     color="Completion", color_continuous_scale='Blues')
    
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()