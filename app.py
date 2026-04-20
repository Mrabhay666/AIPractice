import streamlit as st
import plotly.express as px
from auth_manager import AuthManager
from processor import DataProcessor
from ai_engine import AIEngine

# 1. Page Configuration (Must be the VERY FIRST streamlit command)
st.set_page_config(page_title="AI EDA Pro", layout="wide")

# 2. Custom CSS for Modern "Card" Look
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #FF4B4B; color: white; }
    .auth-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        max-width: 450px;
        margin: auto;
    }
    </style>
""", unsafe_allow_html=True)

auth = AuthManager()

# --- AUTHENTICATION UI ---
if not st.session_state.get('authenticated_user'):
    # Centering the login card
    _, col2, _ = st.columns([1, 2, 1])
    
    with col2:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>🚀 AI Data Explorer</h2>", unsafe_allow_html=True)
        
        mode = st.tabs(["Login", "Create Account"])
        
        with mode[0]: # Login
            u = st.text_input("Username", key="login_u")
            p = st.text_input("Password", type="password", key="login_p")
            if st.button("Sign In"):
                if auth.login(u, p): 
                    st.rerun()
                else: 
                    st.error("Invalid Username or Password")
                
        with mode[1]: # Register
            new_u = st.text_input("New Username")
            new_p = st.text_input("New Password", type="password")
            groq_key = st.text_input("Groq API Key", type="password", help="Saved securely for your sessions")
            
            if st.button("Register & Save Key"):
                if auth.register(new_u, new_p, groq_key):
                    st.success("Account created! Go to Login tab.")
                else:
                    st.warning("All fields are required.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# --- MAIN APPLICATION UI (Only visible if logged in) ---
else:
    # Sidebar Logout and Info
    with st.sidebar:
        st.title(f"👋 Hello, {st.session_state.authenticated_user}!")
        st.info("Your Groq API Key is active.")
        if st.button("Logout"):
            auth.logout()
            st.rerun()

    st.title("📊 Automated EDA Report Generator")
    st.markdown("Upload your dataset and let Groq AI explain the insights.")

    uploaded_file = st.file_uploader("Upload Dataset (CSV/Excel)", type=['csv', 'xlsx'])

    if uploaded_file:
        # Automatically use the saved API key from the user's profile
        api_key = auth.get_api_key()
        
        # 1. Processing Data
        processor = DataProcessor()
        df = processor.load_data(uploaded_file)
        stats = processor.get_summary_stats(df)
        anomalies = processor.identify_anomalies(df)
        
        st.success(f"Data Loaded: {df.shape[0]} rows and {df.shape[1]} columns.")

        # 2. Results Tabs
        tab1, tab2, tab3 = st.tabs(["🤖 AI Narrative Report", "📈 Visual Analytics", "📋 Raw Statistics"])
        
        with tab1:
            st.subheader("GenAI Analysis")
            with st.spinner("Analyzing data trends..."):
                engine = AIEngine(api_key)
                report = engine.generate_narrative_report(stats, anomalies)
                st.markdown(report)
        
        with tab2:
            st.subheader("📊 Interactive Dashboard")
            
            # Row 1: Key Metrics (KPIs)
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            with kpi1:
                st.metric("Total Rows", df.shape[0])
            with kpi2:
                st.metric("Total Columns", df.shape[1])
            with kpi3:
                # Example calculation: finds the first numeric column and sums it
                num_col = df.select_dtypes(include=['number']).columns[0]
                st.metric(f"Total {num_col}", f"{df[num_col].sum():,.0f}")
            with kpi4:
                st.metric("Missing Values", sum(stats['missing_values'].values()))

            st.markdown("---")

            # Row 2: Main Charts
            col_left, col_right = st.columns([1, 1])
            
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            
            with col_left:
                st.markdown("**Distribution Analysis**")
                target = st.selectbox("Select Feature", numeric_cols, key="dist_box")
                fig_hist = px.histogram(df, x=target, nbins=30, marginal="rug", 
                                        color_discrete_sequence=['#0078D4']) # PowerBI Blue
                fig_hist.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350)
                st.plotly_chart(fig_hist, use_container_width=True)

            with col_right:
                st.markdown("**Correlation Heatmap**")
                if len(numeric_cols) > 1:
                    fig_corr = px.imshow(df[numeric_cols].corr(), text_auto=True, 
                                        color_continuous_scale='RdBu_r')
                    fig_corr.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=350)
                    st.plotly_chart(fig_corr, use_container_width=True)

        with tab3:
            st.subheader("Data Profiling & Metadata")
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Missing Values per Column:**")
                st.json(stats['missing_values'])
            with col_b:
                st.write("**Data Types:**")
                st.json(stats['data_types'])
            
            st.write("**Data Preview (Top 10 Rows):**")
            st.dataframe(df.head(10), use_container_width=True)