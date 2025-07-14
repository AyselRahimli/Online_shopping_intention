import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Online Shoppers Purchasing Intention EDA",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .stSelectbox > div > div {
        background-color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🛒 Online Shoppers Purchasing Intention EDA</h1>', unsafe_allow_html=True)

# Load sample data or allow user to upload
@st.cache_data
def load_sample_data():
    """Create sample data that mimics the UCI Online Shoppers dataset structure"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample data based on UCI dataset structure
    data = {
        'Administrative': np.random.poisson(2, n_samples),
        'Administrative_Duration': np.random.exponential(80, n_samples),
        'Informational': np.random.poisson(0.5, n_samples),
        'Informational_Duration': np.random.exponential(25, n_samples),
        'ProductRelated': np.random.poisson(32, n_samples),
        'ProductRelated_Duration': np.random.exponential(1200, n_samples),
        'BounceRates': np.random.uniform(0, 0.2, n_samples),
        'ExitRates': np.random.uniform(0, 0.2, n_samples),
        'PageValues': np.random.exponential(5, n_samples),
        'SpecialDay': np.random.uniform(0, 1, n_samples),
        'Month': np.random.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], n_samples),
        'OperatingSystems': np.random.randint(1, 9, n_samples),
        'Browser': np.random.randint(1, 14, n_samples),
        'Region': np.random.randint(1, 10, n_samples),
        'TrafficType': np.random.randint(1, 21, n_samples),
        'VisitorType': np.random.choice(['New_Visitor', 'Returning_Visitor', 'Other'], 
                                       n_samples, p=[0.1, 0.85, 0.05]),
        'Weekend': np.random.choice([True, False], n_samples, p=[0.23, 0.77]),
        'Revenue': np.random.choice([True, False], n_samples, p=[0.15, 0.85])
    }
    
    return pd.DataFrame(data)

# Sidebar for data loading
st.sidebar.header("📊 Data Loading")
data_source = st.sidebar.radio(
    "Choose data source:",
    ["Sample Data", "Upload CSV"]
)

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully!")
    else:
        st.sidebar.info("Please upload a CSV file or use sample data")
        df = load_sample_data()
else:
    df = load_sample_data()
    st.sidebar.info("Using sample data that mimics the UCI dataset structure")

# Display dataset info
st.markdown('<h2 class="sub-header">📋 Dataset Overview</h2>', unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", len(df))
with col2:
    st.metric("Features", len(df.columns))
with col3:
    st.metric("Numerical Features", len(df.select_dtypes(include=[np.number]).columns))
with col4:
    st.metric("Categorical Features", len(df.select_dtypes(include=['object', 'bool']).columns))

# Display first few rows
if st.checkbox("Show raw data"):
    st.dataframe(df.head())

# Feature information
st.markdown('<h2 class="sub-header">🔍 Feature Information</h2>', unsafe_allow_html=True)

feature_info = {
    'Administrative': 'Number of administrative pages visited',
    'Administrative_Duration': 'Time spent on administrative pages (seconds)',
    'Informational': 'Number of informational pages visited',
    'Informational_Duration': 'Time spent on informational pages (seconds)',
    'ProductRelated': 'Number of product-related pages visited',
    'ProductRelated_Duration': 'Time spent on product-related pages (seconds)',
    'BounceRates': 'Average bounce rate of pages visited',
    'ExitRates': 'Average exit rate of pages visited',
    'PageValues': 'Average page value of pages visited',
    'SpecialDay': 'Closeness to special day (0-1)',
    'Month': 'Month of the year',
    'OperatingSystems': 'Operating system identifier',
    'Browser': 'Browser identifier',
    'Region': 'Region identifier',
    'TrafficType': 'Traffic type identifier',
    'VisitorType': 'Type of visitor',
    'Weekend': 'Whether the session was on weekend',
    'Revenue': 'Whether the session ended in a transaction (target)'
}

selected_feature = st.selectbox("Select a feature to see its description:", list(feature_info.keys()))
st.info(f"**{selected_feature}**: {feature_info[selected_feature]}")

# Sidebar for EDA options
st.sidebar.header("🔧 EDA Options")
analysis_type = st.sidebar.selectbox(
    "Choose analysis type:",
    ["Basic Statistics", "Distribution Analysis", "Correlation Analysis", 
     "Target Analysis", "Time Series Analysis", "Model Exploration"]
)

# Main analysis section
st.markdown(f'<h2 class="sub-header">📊 {analysis_type}</h2>', unsafe_allow_html=True)

if analysis_type == "Basic Statistics":
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Numerical Features Statistics")
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        st.dataframe(df[numerical_cols].describe())
    
    with col2:
        st.subheader("Categorical Features Summary")
        categorical_cols = df.select_dtypes(include=['object', 'bool']).columns
        for col in categorical_cols:
            st.write(f"**{col}**:")
            st.write(df[col].value_counts().head())
            st.write("---")

elif analysis_type == "Distribution Analysis":
    feature_to_plot = st.selectbox(
        "Select feature to analyze:",
        df.columns.tolist()
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if df[feature_to_plot].dtype in ['object', 'bool']:
            fig = px.bar(
                x=df[feature_to_plot].value_counts().index,
                y=df[feature_to_plot].value_counts().values,
                title=f"Distribution of {feature_to_plot}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.histogram(
                df, 
                x=feature_to_plot, 
                title=f"Distribution of {feature_to_plot}",
                nbins=30
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if df[feature_to_plot].dtype not in ['object', 'bool']:
            fig = px.box(
                df, 
                y=feature_to_plot,
                title=f"Box Plot of {feature_to_plot}"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            # Show pie chart for categorical variables
            fig = px.pie(
                values=df[feature_to_plot].value_counts().values,
                names=df[feature_to_plot].value_counts().index,
                title=f"Pie Chart of {feature_to_plot}"
            )
            st.plotly_chart(fig, use_container_width=True)

elif analysis_type == "Correlation Analysis":
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numerical_cols) > 1:
        correlation_matrix = df[numerical_cols].corr()
        
        fig = px.imshow(
            correlation_matrix,
            title="Correlation Matrix of Numerical Features",
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature correlation with target
        if 'Revenue' in df.columns:
            # Convert Revenue to numeric for correlation
            df_temp = df.copy()
            df_temp['Revenue'] = df_temp['Revenue'].astype(int)
            
            target_corr = df_temp[numerical_cols + ['Revenue']].corr()['Revenue'].sort_values(ascending=False)
            
            fig = px.bar(
                x=target_corr.index[1:],  # Exclude Revenue itself
                y=target_corr.values[1:],
                title="Feature Correlation with Revenue",
                color=target_corr.values[1:],
                color_continuous_scale="RdYlBu"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Need at least 2 numerical features for correlation analysis")

elif analysis_type == "Target Analysis":
    if 'Revenue' in df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            revenue_counts = df['Revenue'].value_counts()
            fig = px.pie(
                values=revenue_counts.values,
                names=revenue_counts.index,
                title="Revenue Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Revenue Statistics")
            revenue_stats = df['Revenue'].value_counts()
            st.write(f"Total Sessions: {len(df)}")
            st.write(f"Purchased: {revenue_stats.get(True, 0)} ({revenue_stats.get(True, 0)/len(df)*100:.1f}%)")
            st.write(f"Not Purchased: {revenue_stats.get(False, 0)} ({revenue_stats.get(False, 0)/len(df)*100:.1f}%)")
        
        # Feature comparison by Revenue
        st.subheader("Feature Analysis by Revenue")
        feature_for_comparison = st.selectbox(
            "Select feature to compare by Revenue:",
            [col for col in df.columns if col != 'Revenue']
        )
        
        if df[feature_for_comparison].dtype in ['object', 'bool']:
            comparison_data = df.groupby([feature_for_comparison, 'Revenue']).size().reset_index(name='Count')
            fig = px.bar(
                comparison_data,
                x=feature_for_comparison,
                y='Count',
                color='Revenue',
                title=f"{feature_for_comparison} vs Revenue",
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            fig = px.box(
                df,
                x='Revenue',
                y=feature_for_comparison,
                title=f"{feature_for_comparison} Distribution by Revenue"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Revenue column not found in the dataset")

elif analysis_type == "Time Series Analysis":
    if 'Month' in df.columns:
        month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        # Monthly sessions
        monthly_sessions = df['Month'].value_counts().reindex(month_order, fill_value=0)
        
        fig = px.line(
            x=monthly_sessions.index,
            y=monthly_sessions.values,
            title="Sessions by Month",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Monthly revenue if available
        if 'Revenue' in df.columns:
            monthly_revenue = df[df['Revenue'] == True]['Month'].value_counts().reindex(month_order, fill_value=0)
            monthly_conversion = (monthly_revenue / monthly_sessions * 100).fillna(0)
            
            fig = px.bar(
                x=monthly_conversion.index,
                y=monthly_conversion.values,
                title="Conversion Rate by Month (%)",
                color=monthly_conversion.values,
                color_continuous_scale="viridis"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Weekend analysis
        if 'Weekend' in df.columns:
            weekend_analysis = df.groupby('Weekend').size()
            fig = px.bar(
                x=['Weekday', 'Weekend'],
                y=[weekend_analysis.get(False, 0), weekend_analysis.get(True, 0)],
                title="Sessions: Weekday vs Weekend"
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Month column not found in the dataset")

elif analysis_type == "Model Exploration":
    st.subheader("Quick Model Training")
    
    if 'Revenue' in df.columns:
        # Prepare data for modeling
        df_model = df.copy()
        
        # Handle categorical variables
        categorical_cols = df_model.select_dtypes(include=['object']).columns
        le_dict = {}
        
        for col in categorical_cols:
            le = LabelEncoder()
            df_model[col] = le.fit_transform(df_model[col].astype(str))
            le_dict[col] = le
        
        # Convert boolean to int
        bool_cols = df_model.select_dtypes(include=['bool']).columns
        for col in bool_cols:
            df_model[col] = df_model[col].astype(int)
        
        # Prepare features and target
        X = df_model.drop('Revenue', axis=1)
        y = df_model['Revenue']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model selection
        model_type = st.selectbox(
            "Select model type:",
            ["Random Forest", "Logistic Regression"]
        )
        
        if st.button("Train Model"):
            if model_type == "Random Forest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                fig = px.bar(
                    feature_importance.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title="Top 10 Feature Importance (Random Forest)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                model = LogisticRegression(random_state=42)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            
            # Model performance
            accuracy = accuracy_score(y_test, y_pred)
            st.success(f"Model Accuracy: {accuracy:.3f}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig = px.imshow(
                cm,
                title="Confusion Matrix",
                labels=dict(x="Predicted", y="Actual"),
                x=['No Purchase', 'Purchase'],
                y=['No Purchase', 'Purchase'],
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Classification report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose())
    else:
        st.warning("Revenue column not found for model training")

# Sidebar adjustments
st.sidebar.header("🎛️ Adjustments")
if st.sidebar.checkbox("Show advanced options"):
    st.sidebar.subheader("Data Filtering")
    
    # Filter by visitor type
    if 'VisitorType' in df.columns:
        visitor_types = st.sidebar.multiselect(
            "Filter by Visitor Type:",
            df['VisitorType'].unique(),
            default=df['VisitorType'].unique()
        )
        df = df[df['VisitorType'].isin(visitor_types)]
    
    # Filter by weekend
    if 'Weekend' in df.columns:
        weekend_filter = st.sidebar.radio(
            "Filter by Weekend:",
            ["All", "Weekday", "Weekend"]
        )
        if weekend_filter == "Weekday":
            df = df[df['Weekend'] == False]
        elif weekend_filter == "Weekend":
            df = df[df['Weekend'] == True]
    
    # Filter by month
    if 'Month' in df.columns:
        selected_months = st.sidebar.multiselect(
            "Filter by Month:",
            df['Month'].unique(),
            default=df['Month'].unique()
        )
        df = df[df['Month'].isin(selected_months)]

# Footer
st.markdown("---")
st.markdown("""
**Data Source**: Sakar, C. & Kastro, Y. (2018). Online Shoppers Purchasing Intention Dataset. UCI Machine Learning Repository.

**Features**:
- 📊 Interactive visualizations with Plotly
- 🔍 Comprehensive EDA capabilities
- 🤖 Basic machine learning exploration
- 🎛️ Dynamic filtering options
- 📈 Time series analysis
- 🎯 Target variable analysis
""")
