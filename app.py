import streamlit as st
import numpy as np
import joblib
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import time
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")

model = joblib.load("best_model.pkl")

#st.dataframe(model)
#st.title("GradeInsight Students Grades Predictor")

df = joblib.load("dataframe.pkl")

# Config Settings
st.set_page_config(
        page_title="GradeInsight",
        layout="wide",
        initial_sidebar_state="collapsed"
)


# ── CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem; border-radius: 12px; margin-bottom: 1.5rem; text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 2.5rem; margin: 0; }
    .main-header p  { color: #a8b2c1; margin: 0.5rem 0 0; }
    .metric-card {
        background: #1a1a2e; border: 1px solid #0f3460;
        border-radius: 10px; padding: 1.2rem; text-align: center;
    }
    .metric-card h2 { color: #e94560; font-size: 2rem; margin: 0; }
    .metric-card p  { color: #a8b2c1; margin: 0; font-size: 0.85rem; }
    .grade-badge {
        display: inline-block; padding: 0.4rem 1.2rem;
        border-radius: 25px; font-size: 1.8rem; font-weight: bold;
    }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# Header
st.markdown("""
<div class="main-header">
    <h1>GradeInsight</h1>
    <p>Predict student performance based on socio-economic and lifestyle factors</p>
</div>
""", unsafe_allow_html=True)


# SideBar

with st.sidebar:
    st.image("graduate-icon.png",width=80)
    st.title("Navigation")

    page = st.radio("",[
        "Dashboard",
        "EDA",
        "Predict Grade",
        "Batch Prediction",
        "Model Performance",
    ], label_visibility="collapsed")
    
    # st.divider()
    # st.markdown("**Data Source**")
    #uploaded = st.file_uploader("Upload CSV", type=["csv"])

    # st.divider()


    st.markdown("**Filters**")
    if "gender" in df.columns:
        gender = ["All"] + sorted(df["gender"].dropna().unique().tolist())
        sel_gender = st.selectbox("Gender", gender)
    else:
        sel_gender = "All"

    if sel_gender != "All" and "gender" in df.columns:
        df = df[df["gender"] == sel_gender]



    
numeric_features = ['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
       'attendance_percentage', 'sleep_hours', 'exercise_frequency',
       'mental_health_rating']


if page == "Dashboard":
    st.subheader("Overview")

    c1,c2,c3,c4,c5 = st.columns(5)

    kpis = {
        "Total Students" : f"{len(df):,}",
        "Average Attendence" : f"{df['attendance_percentage'].mean():.1f}%" if "attendance_percentage" in df.columns else "—",
        "Average Sleep Hours" : f"{df['sleep_hours'].mean():.1f}" if "sleep_hours" in df.columns else "-",
        "Average Study Hours/Day" : f"{df['study_hours_per_day'].mean():.1f}" if "study_hours_per_day" in df.columns else "-",
        "Passing Rate" : f"{df['exam_score'].mean():.2f}%" if "exam_score" in df.columns else "-",

    }
    for col,(label, val) in zip([c1,c2,c3,c4,c5], kpis.items()):
                col.markdown(f"""
        <div class="metric-card">
            <h2>{val}</h2><p>{label}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)
            

    #fig,ax =plt.subplots(figsize=(4,3))
    #sns.countplot(x="gender",data=df,ax=ax)
    #st.pyplot(fig,use_container_width=False)

    cat_col = ["gender","part_time_job","diet_quality","parental_education_level",
                "internet_quality","extracurricular_participation"]

    cols = st.columns(2)  # 2 plots per row

    for i, col in enumerate(cat_col):
        with cols[i % 2]:
            st.subheader(col)

            fig, ax = plt.subplots(figsize=(4,3))
            sns.countplot(x=col, data=df, ax=ax)

            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig, use_container_width=False)

elif page == "EDA":
    st.subheader("Exploratory Data Analysis")

    tab1,tab2,tab3 = st.tabs(["Summary Stats","Missing Values","Distributions"])

    with tab1:
            col_a, col_b = st.columns(2)
            col_a.metric("Rows",f"{df.shape[0]:,}")
            col_b.metric("Columns", f"{df.shape[1]}")
            st.dataframe(df.describe(include=[np.number]).round(2), use_container_width=True)

    with tab2:
            miss = df.isnull().sum().reset_index()
            miss.columns = ["Feature","Missing"]
            miss = miss[miss["Missing"]>0].sort_values("Missing", ascending=False)
            if miss.empty:
                    st.success("No Missing Values After Imputation !")
            else:
                    fig = px.bar(miss, x="Feature", y="Missing",
                    title="Missing Values per Feature", template="plotly_dark", color="Missing",color_continuous_scale="Reds")
                    st.plotly_chart(fig, use_container_width=True)




    with tab3:
            feat = st.selectbox("Select Feature",
                                [c for c in numeric_features if c in df.columns])
            
            fig = make_subplots(rows=1, cols=2,subplot_titles=("Histogram","Box Plot"))
            
            fig.add_trace(
                    go.Histogram(x=df[feat],name=feat,
                                marker_color="#4C72B0",opacity=0.8),
                                row=1,col=1,
            )

            fig.add_trace(
            go.Box(y=df[feat], name=feat, marker_color="#4C72B0"),
                row=1, col=2,
                            )
            
            fig.update_layout(template="plotly_dark",
                            title=f"Distribution of {feat}",
                            showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

# correlation heatmap

    corr = df.corr(numeric_only=True)

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        title="Correlation Heatmap",
    )

    st.plotly_chart(fig, use_container_width=True)

# Scatter Plot
    st.subheader("Scatter Plot")
    cols = st.columns(2)

    for i, col in enumerate(numeric_features):
        fig = px.scatter(
            df,
            x=col,
            y="exam_score",
            trendline="ols",
            title=f"{col} vs Exam Score"
        )
        with cols[i % 2]:
                st.plotly_chart(fig, use_container_width=True)






elif page == "Predict Grade":
        st.subheader("Predict Grade")

                # ── Global CSS ──────────────────────────────────────────────────────────────────
        st.markdown("""
        <style>
        /* ── Google Fonts ── */
        @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

        /* ── Root variables ── */
        :root {
            --surface:   #0e1420;
            --card:      #111827;
            --border:    rgba(99,202,255,0.12);
            --glow:      #63caff;
            --accent:    #ff6b6b;
            --gold:      #ffd166;
            --green:     #06d6a0;
            --text:      #030303;
            --muted:     #6b7c93;
            --radius:    16px;
        }

        /* ── Base resets ── */
        html, body, [class*="css"] {
            background-color: var(--bg) !important;
            color: var(--text) !important;
            font-family: 'Syne', sans-serif !important;
        }

        .stApp { background: var(--bg) !important; }
        #MainMenu, footer, header { visibility: hidden; }
        .block-container { padding: 2rem 3rem !important; max-width: 1200px !important; }

        /* ── Scrollbar ── */
        ::-webkit-scrollbar { width: 6px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

        /* ── Hero header ── */
        .hero {
            background: linear-gradient(135deg, #0d1b2a 0%, #0e2040 50%, #0d1b2a 100%);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 2.5rem 3rem;
            margin-bottom: 2.5rem;
            position: relative;
            overflow: hidden;
        }
        .hero::before {
            content: '';
            position: absolute; inset: 0;
            background:
                radial-gradient(ellipse at 20% 50%, rgba(99,202,255,0.08) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 20%, rgba(255,107,107,0.06) 0%, transparent 50%);
            pointer-events: none;
        }
        .hero-tag {
            display: inline-block;
            background: rgba(99,202,255,0.1);
            border: 1px solid rgba(99,202,255,0.3);
            color: var(--glow);
            font-family: 'Space Mono', monospace;
            font-size: 0.7rem;
            letter-spacing: 0.15em;
            padding: 4px 12px;
            border-radius: 20px;
            margin-bottom: 1rem;
            text-transform: uppercase;
        }
        .hero-title {
            font-size: clamp(2rem, 4vw, 3.2rem);
            font-weight: 800;
            line-height: 1.1;
            margin: 0 0 0.6rem;
            background: linear-gradient(90deg, #ffffff 0%, var(--glow) 60%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .hero-sub {
            font-size: 1rem;
            color: var(--muted);
            max-width: 520px;
            line-height: 1.6;
            font-weight: 400;
        }
        .hero-badge {
            position: absolute;
            right: 3rem; top: 50%;
            transform: translateY(-50%);
            width: 120px; height: 120px;
            border-radius: 50%;
            background: conic-gradient(var(--glow) 0deg, #a78bfa 120deg, var(--accent) 240deg, var(--glow) 360deg);
            display: flex; align-items: center; justify-content: center;
            font-size: 2.8rem;
            box-shadow: 0 0 60px rgba(99,202,255,0.25);
            animation: spin-slow 12s linear infinite;
        }
        .hero-badge-inner {
            width: 100px; height: 100px;
            border-radius: 50%;
            background: var(--surface);
            display: flex; align-items: center; justify-content: center;
            font-size: 2.8rem;
        }
        @keyframes spin-slow { to { transform: translateY(-50%) rotate(360deg); } }

        /* ── Section labels ── */
        .section-label {
            font-family: 'Space Mono', monospace;
            font-size: 0.65rem;
            letter-spacing: 0.2em;
            color: var(--glow);
            text-transform: uppercase;
            margin-bottom: 1.2rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .section-label::after {
            content: '';
            flex: 1;
            height: 1px;
            background: linear-gradient(90deg, var(--border), transparent);
        }

        /* ── Metric cards ── */
        .metric-row {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 1rem;
            margin-bottom: 2rem;
        }
        .metric-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: var(--radius);
            padding: 1.2rem 1.4rem;
            position: relative;
            overflow: hidden;
            transition: border-color 0.3s;
        }
        .metric-card:hover { border-color: rgba(99,202,255,0.3); }
        .metric-card::before {
            content: '';
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 2px;
            background: var(--accent-color, var(--glow));
            border-radius: 2px 2px 0 0;
        }
        .metric-icon {
            font-size: 1.4rem;
            margin-bottom: 0.5rem;
            display: block;
        }
        .metric-label {
            font-size: 0.7rem;
            color: var(--muted);
            text-transform: uppercase;
            letter-spacing: 0.1em;
            font-family: 'Space Mono', monospace;
            margin-bottom: 0.2rem;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 800;
            color: var(--text);
            line-height: 1;
        }
        .metric-unit {
            font-size: 0.75rem;
            color: var(--muted);
            font-weight: 400;
            margin-left: 3px;
        }

        /* ── Input panel ── */
        .panel {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 2rem;
            margin-bottom: 1.5rem;
        }
        .panel-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        /* ── Sliders ── */
        div[data-baseweb="slider"] {
            padding: 0 !important;
        }
        div[data-baseweb="slider"] [data-testid="stTickBar"] { display: none; }
        .stSlider > div > div > div {
            background: rgba(99,202,255,0.15) !important;
            height: 6px !important;
            border-radius: 3px !important;
        }
        .stSlider > div > div > div > div {
            background: linear-gradient(90deg, var(--glow), #a78bfa) !important;
            height: 6px !important;
            border-radius: 3px !important;
        }
        .stSlider [role="slider"] {
            background: #fff !important;
            border: 3px solid var(--glow) !important;
            box-shadow: 0 0 12px rgba(99,202,255,0.5) !important;
            width: 18px !important;
            height: 18px !important;
        }
        .stSlider label {
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            color: var(--muted) !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            font-family: 'Space Mono', monospace !important;
        }

        /* ── Selectbox ── */
        .stSelectbox > div > div {
            background: rgba(99,202,255,0.05) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            color: var(--text) !important;
        }
        .stSelectbox label {
            font-size: 0.78rem !important;
            font-weight: 600 !important;
            color: var(--muted) !important;
            text-transform: uppercase !important;
            letter-spacing: 0.08em !important;
            font-family: 'Space Mono', monospace !important;
        }

        /* ── Button ── */
        .stButton > button {
            width: 100% !important;
            background: linear-gradient(135deg, #1a6fff 0%, #63caff 50%, #a78bfa 100%) !important;
            background-size: 200% 200% !important;
            animation: gradient-shift 3s ease infinite !important;
            color: #fff !important;
            font-family: 'Syne', sans-serif !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            letter-spacing: 0.05em !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 0.9rem 2rem !important;
            cursor: pointer !important;
            box-shadow: 0 4px 30px rgba(99,202,255,0.3) !important;
            transition: transform 0.15s, box-shadow 0.15s !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 8px 40px rgba(99,202,255,0.4) !important;
        }
        @keyframes gradient-shift {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        /* ── Result box ── */
        .result-box {
            background: linear-gradient(135deg, #0d1f12 0%, #081a2b 100%);
            border: 1px solid rgba(6,214,160,0.3);
            border-radius: 20px;
            padding: 2.5rem;
            text-align: center;
            position: relative;
            overflow: hidden;
            animation: fadeSlideUp 0.5s ease;
        }
        .result-box::before {
            content: '';
            position: absolute; inset: 0;
            background: radial-gradient(ellipse at 50% 0%, rgba(6,214,160,0.08) 0%, transparent 70%);
        }
        .result-score {
            font-size: clamp(4rem, 8vw, 7rem);
            font-weight: 800;
            line-height: 1;
            background: linear-gradient(135deg, var(--green) 0%, #a8edda 50%, var(--gold) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: none;
            margin-bottom: 0.5rem;
        }
        .result-grade {
            display: inline-block;
            font-family: 'Space Mono', monospace;
            font-size: 1rem;
            background: rgba(6,214,160,0.15);
            border: 1px solid rgba(6,214,160,0.4);
            color: var(--green);
            padding: 4px 18px;
            border-radius: 20px;
            margin-bottom: 1rem;
            letter-spacing: 0.1em;
        }
        .result-label {
            font-size: 0.8rem;
            color: var(--muted);
            font-family: 'Space Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 0.15em;
        }
        .result-bar-wrap {
            margin: 1.5rem auto 0;
            max-width: 400px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            height: 8px;
            overflow: hidden;
        }
        .result-bar-fill {
            height: 100%;
            border-radius: 8px;
            background: linear-gradient(90deg, var(--green), var(--gold));
            animation: grow 1s ease-out;
            transform-origin: left;
        }
        @keyframes grow { from { width: 0; } }
        @keyframes fadeSlideUp {
            from { opacity: 0; transform: translateY(20px); }
            to   { opacity: 1; transform: translateY(0); }
        }

        /* ── Progress rings ── */
        .ring-row {
            display: flex;
            gap: 1.2rem;
            justify-content: center;
            flex-wrap: wrap;
            margin-top: 1.5rem;
        }
        .ring-item { text-align: center; }
        .ring-item svg { display: block; margin: 0 auto 4px; }
        .ring-label { font-size: 0.65rem; color: var(--muted); font-family: 'Space Mono', monospace; letter-spacing: 0.08em; text-transform: uppercase; }

        /* ── Divider ── */
        .fancy-divider {
            display: flex; align-items: center; gap: 1rem;
            margin: 2rem 0;
            color: var(--muted);
            font-size: 0.7rem;
            font-family: 'Space Mono', monospace;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }
        .fancy-divider::before, .fancy-divider::after {
            content: '';
            flex: 1;
            height: 1px;
            background: var(--border);
        }

        /* ── Tips card ── */
        .tip-card {
            background: rgba(255,209,102,0.05);
            border: 1px solid rgba(255,209,102,0.2);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin-top: 1rem;
            font-size: 0.85rem;
            color: var(--gold);
            display: flex;
            gap: 10px;
            align-items: flex-start;
            line-height: 1.5;
        }
        .tip-icon { font-size: 1rem; flex-shrink: 0; margin-top: 1px; }
        </style>
        """, unsafe_allow_html=True)
            
        

        study_hours = st.slider("Study Hours Per Day",0.0,12.0,2.0)
        attendance = st.slider("Attendance Percentage",0.0,100.0,80.0)
        sleep_hours = st.slider("Sleep Hours Per Night",0.0,12.0,7.0)
        part_time_job = st.selectbox("Part-Time Job",["No","Yes"] )
        mental_health = st.slider("Mental Health Rating(1-10)",0.0,10.0,5.0)
        ptj_encoded = 1 if part_time_job == "Yes" else 0

        if st.button("Predict Exam Score"):

            with st.spinner("Running Prediction Model ..."):
                    time.sleep(0.6)



                    input_data = np.array([[study_hours,attendance,mental_health,sleep_hours,ptj_encoded]])
                    prediction = model.predict(input_data)[0]

                    prediction = max(0, min(100,prediction))

                    #st.success(f"Predicted Grade Score: {prediction:.2f}")

                    if prediction >=90:
                        grade, grade_color = "A+ Outstanding","#06d6a0"
                    elif prediction >= 80:
                        grade, grade_color = "A  · Excellent", "#06d6a0"
                    elif prediction >= 70:
                        grade, grade_color = "B  · Good", "#63caff"
                    elif prediction >= 60:
                        grade, grade_color = "C  · Average", "#ffd166"
                    elif prediction >= 50:
                        grade, grade_color = "D  · Below Avg", "#ff9f43"
                    else:
                        grade, grade_color = "F  · Needs Work", "#ff6b6b"    

                    # SVG ring helper
                    def ring_svg(value, max_val, color, label, size=64, stroke=5):
                        r = (size - stroke * 2) / 2
                        circ = 2 * 3.14159 * r
                        dash = (value / max_val) * circ
                        return f"""
                        <div class="ring-item">
                            <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
                            <circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none"
                                    stroke="rgba(255,255,255,0.06)" stroke-width="{stroke}"/>
                            <circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none"
                                    stroke="{color}" stroke-width="{stroke}"
                                    stroke-dasharray="{dash:.1f} {circ:.1f}"
                                    stroke-linecap="round"
                                    transform="rotate(-90 {size/2} {size/2})"/>
                            <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle"
                                    fill="{color}" font-size="10" font-family="Space Mono,monospace" font-weight="700">
                                {value:.0f}
                            </text>
                            </svg>
                            <div class="ring-label">{label}</div>
                        </div>"""    
                    rings = (
                            ring_svg(study_hours, 12, "#63caff", "Study") +
                            ring_svg(attendance, 100, "#a78bfa", "Attend") +
                            ring_svg(sleep_hours, 12, "#ffd166", "Sleep") +
                            ring_svg(mental_health, 10, "#06d6a0", "Wellness")
                        )
            st.markdown(f"""
        <div class="fancy-divider">Prediction Result</div>
        <div class="result-box">
            <div class="result-label">Predicted Exam Score</div>
            <div class="result-score">{prediction:.1f}</div>
            <div class="result-grade" style="border-color:{grade_color}44;background:{grade_color}18;color:{grade_color};">{grade}</div>
            <div class="result-bar-wrap">
                <div class="result-bar-fill" style="width:{prediction:.1f}%"></div>
            </div>
            <div class="ring-row">{rings}</div>
        </div>
        """, unsafe_allow_html=True)


        else:
                    st.markdown("""
                        <div style="
                            border: 1px dashed rgba(99,202,255,0.15);
                            border-radius: 20px;
                            padding: 3rem 2rem;
                            text-align: center;
                            margin-top: 0.5rem;
                        ">
                            <div style="font-size:3rem;margin-bottom:1rem;opacity:0.5;">⚡</div>
                            <div style="color:var(--muted);font-family:'Space Mono',monospace;font-size:0.78rem;
                                        letter-spacing:0.1em;text-transform:uppercase;">
                                Set your parameters<br>and hit Predict
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

elif page == "Batch Prediction":
    st.subheader("Coming Soon ...")

    st.info("Upload a CSV file with student features to get bulk predictions.")
    batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"])

    if batch_file is not None:
          
        batch_df = pd.read_csv(batch_file)
        if('exam_score') in batch_df.columns:
            batch_df.drop(columns=['exam_score'])

        batch_prediction = model.predict(batch_df)
        batch_df['predictions_new'] = batch_prediction
        st.write("### Predictions", batch_prediction.head())

        # Download button
        csv = batch_prediction.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name='batch_predictions.csv',
            mime='text/csv'
        )
    else:
            st.warning("Please Upload a CSV file first")    






elif page == "Model Performance":
    st.subheader("Model Performance")

    df1 = joblib.load("results_model.pkl")

    table1, = st.tabs(["Confusion Matrix"])
    with table1:
           # col_a, col_b = st.columns(2)
           # col_a.metric("Rows",f"{df1.shape[0]:,}")
           # col_b.metric("Columns", f"{df1.shape[1]}")
            st.dataframe(df1, use_container_width=True)

    
    fig3 = px.bar(df1,x="model",y="r2",
    title="Model Comparison",
    labels={"model": "Models", "r2": "R² Score"}
)

    st.plotly_chart(fig3, use_container_width=True)            



# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2.5rem 0 1rem;color:var(--muted);
            font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:0.12em;">
    Created By Sahal Fitter
</div>
""", unsafe_allow_html=True)


