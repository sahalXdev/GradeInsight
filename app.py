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
import io
from plotly.subplots import make_subplots
warnings.filterwarnings("ignore")

model = joblib.load("best_model.pkl")
df = joblib.load("dataframe.pkl")

# ── MODEL FEATURE CONFIG ──────────────────────────────────────────────────────
# These are the features the model was trained on
MODEL_FEATURES = ['study_hours_per_day', 'attendance_percentage', 'mental_health_rating',
                  'sleep_hours', 'part_time_job']

FEATURE_DEFAULTS = {
    'study_hours_per_day':    2.0,
    'attendance_percentage':  75.0,
    'mental_health_rating':   5.0,
    'sleep_hours':            7.0,
    'part_time_job':          0,
}

FEATURE_LABELS = {
    'study_hours_per_day':   'Study Hours / Day',
    'attendance_percentage': 'Attendance %',
    'mental_health_rating':  'Mental Health (1-10)',
    'sleep_hours':           'Sleep Hours',
    'part_time_job':         'Part-Time Job (0/1)',
}

def encode_batch_df(raw_df):
    """Encode a raw uploaded dataframe for the model, filling missing features with defaults."""
    out = pd.DataFrame()
    for feat in MODEL_FEATURES:
        if feat in raw_df.columns:
            col = raw_df[feat].copy()
            # encode yes/no strings for part_time_job
            if feat == 'part_time_job':
                col = col.astype(str).str.strip().str.lower()
                col = col.map({'yes': 1, 'no': 0, '1': 1, '0': 0, 'true': 1, 'false': 0})
                col = col.fillna(FEATURE_DEFAULTS[feat])
            col = pd.to_numeric(col, errors='coerce').fillna(FEATURE_DEFAULTS[feat])
            out[feat] = col
        else:
            out[feat] = FEATURE_DEFAULTS[feat]
    return out

def score_to_grade(score):
    if score >= 90: return "A+", "#06d6a0"
    elif score >= 80: return "A", "#06d6a0"
    elif score >= 70: return "B", "#63caff"
    elif score >= 60: return "C", "#ffd166"
    elif score >= 50: return "D", "#ff9f43"
    else: return "F", "#ff6b6b"

# ── Config Settings ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GradeInsight",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
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
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 500; }

    /* Bulk Scanner styles */
    .feature-tag {
        display:inline-block; padding:3px 10px; border-radius:20px;
        font-size:0.78rem; font-weight:600; margin:2px;
    }
    .tag-found   { background:#06d6a020; border:1px solid #06d6a0; color:#06d6a0; }
    .tag-missing { background:#ff6b6b20; border:1px solid #ff6b6b; color:#ff6b6b; }
    .tag-imputed { background:#ffd16620; border:1px solid #ffd166; color:#ffd166; }
    .scanner-card {
        background:#111827; border:1px solid rgba(99,202,255,0.15);
        border-radius:16px; padding:1.5rem; margin-bottom:1rem;
    }
    .scanner-title { color:#63caff; font-size:1rem; font-weight:700; margin-bottom:0.8rem; }
    .stat-pill {
        display:inline-block; padding:4px 14px; border-radius:20px;
        font-size:0.85rem; font-weight:600; margin:4px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>GradeInsight</h1>
    <p>Predict student performance based on socio-economic and lifestyle factors</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("graduate-icon.png", width=80)
    st.title("Navigation")
    page = st.radio("", [
        "Dashboard",
        "EDA",
        "Predict Grade",
        "Bulk Scanner",
        "Model Performance",
    ], label_visibility="collapsed")

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


# ══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.subheader("Overview")
    c1, c2, c3, c4, c5 = st.columns(5)
    kpis = {
        "Total Students":         f"{len(df):,}",
        "Avg Attendance":         f"{df['attendance_percentage'].mean():.1f}%" if "attendance_percentage" in df.columns else "—",
        "Avg Sleep Hours":        f"{df['sleep_hours'].mean():.1f}" if "sleep_hours" in df.columns else "-",
        "Avg Study Hours/Day":    f"{df['study_hours_per_day'].mean():.1f}" if "study_hours_per_day" in df.columns else "-",
        "Avg Exam Score":         f"{df['exam_score'].mean():.2f}" if "exam_score" in df.columns else "-",
    }
    for col, (label, val) in zip([c1, c2, c3, c4, c5], kpis.items()):
        col.markdown(f"""
        <div class="metric-card">
            <h2>{val}</h2><p>{label}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cat_col = ["gender", "part_time_job", "diet_quality", "parental_education_level",
               "internet_quality", "extracurricular_participation"]
    cols = st.columns(2)
    for i, col in enumerate(cat_col):
        with cols[i % 2]:
            st.subheader(col)
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.countplot(x=col, data=df, ax=ax)
            plt.xticks(rotation=45)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)


# ══════════════════════════════════════════════════════════════════════════════
#  EDA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA":
    st.subheader("Exploratory Data Analysis")
    tab1, tab2, tab3 = st.tabs(["Summary Stats", "Missing Values", "Distributions"])

    with tab1:
        col_a, col_b = st.columns(2)
        col_a.metric("Rows", f"{df.shape[0]:,}")
        col_b.metric("Columns", f"{df.shape[1]}")
        st.dataframe(df.describe(include=[np.number]).round(2), use_container_width=True)

    with tab2:
        miss = df.isnull().sum().reset_index()
        miss.columns = ["Feature", "Missing"]
        miss = miss[miss["Missing"] > 0].sort_values("Missing", ascending=False)
        if miss.empty:
            st.success("No Missing Values After Imputation!")
        else:
            fig = px.bar(miss, x="Feature", y="Missing",
                         title="Missing Values per Feature", template="plotly_dark",
                         color="Missing", color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        feat = st.selectbox("Select Feature", [c for c in numeric_features if c in df.columns])
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Histogram", "Box Plot"))
        fig.add_trace(go.Histogram(x=df[feat], name=feat, marker_color="#4C72B0", opacity=0.8), row=1, col=1)
        fig.add_trace(go.Box(y=df[feat], name=feat, marker_color="#4C72B0"), row=1, col=2)
        fig.update_layout(template="plotly_dark", title=f"Distribution of {feat}", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    corr = df.corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Blues")
    fig.update_layout(title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Scatter Plot")
    cols = st.columns(2)
    for i, col in enumerate(numeric_features):
        fig = px.scatter(df, x=col, y="exam_score", trendline="ols", title=f"{col} vs Exam Score")
        with cols[i % 2]:
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PREDICT GRADE (Single)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Predict Grade":
    st.subheader("Predict Grade")

    study_hours  = st.slider("Study Hours Per Day", 0.0, 12.0, 2.0)
    attendance   = st.slider("Attendance Percentage", 0.0, 100.0, 80.0)
    sleep_hours  = st.slider("Sleep Hours Per Night", 0.0, 12.0, 7.0)
    part_time_job = st.selectbox("Part-Time Job", ["No", "Yes"])
    mental_health = st.slider("Mental Health Rating (1-10)", 0.0, 10.0, 5.0)
    ptj_encoded  = 1 if part_time_job == "Yes" else 0

    if st.button("Predict Exam Score"):
        with st.spinner("Running Prediction Model..."):
            time.sleep(0.6)
            input_data = np.array([[study_hours, attendance, mental_health, sleep_hours, ptj_encoded]])
            prediction = float(model.predict(input_data)[0])
            prediction = max(0, min(100, prediction))
            grade, grade_color = score_to_grade(prediction)

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
        <div class="result-box" style="background:linear-gradient(135deg,#0d1f12 0%,#081a2b 100%);
             border:1px solid rgba(6,214,160,0.3);border-radius:20px;padding:2.5rem;text-align:center;">
            <div style="font-size:0.8rem;color:#6b7c93;text-transform:uppercase;letter-spacing:0.15em;">Predicted Exam Score</div>
            <div style="font-size:5rem;font-weight:800;background:linear-gradient(135deg,#06d6a0,#ffd166);
                 -webkit-background-clip:text;-webkit-text-fill-color:transparent;">{prediction:.1f}</div>
            <div style="display:inline-block;padding:4px 18px;border-radius:20px;border:1px solid {grade_color}44;
                 background:{grade_color}18;color:{grade_color};font-size:1rem;">{grade}</div>
            <div style="max-width:400px;margin:1rem auto 0;background:rgba(255,255,255,0.05);
                 border-radius:8px;height:8px;overflow:hidden;">
                <div style="width:{prediction:.1f}%;height:100%;border-radius:8px;
                     background:linear-gradient(90deg,#06d6a0,#ffd166);"></div>
            </div>
            <div style="display:flex;gap:1.2rem;justify-content:center;flex-wrap:wrap;margin-top:1.5rem;">{rings}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  BULK SCANNER  — automatic feature selection
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Bulk Scanner":
    st.subheader("🔍 Bulk Scanner — Automatic Feature Detection")

    st.markdown("""
    Upload a CSV file with **any subset** of the model features.  
    The scanner automatically detects which features are present, encodes them correctly,
    and fills missing features with smart defaults so predictions always work.
    """)

    # ── Sample CSV download ────────────────────────────────────────────────────
    with st.expander("📄 Download Sample CSV Template"):
        sample_data = {
            'student_id':              ['S2001', 'S2002', 'S2003', 'S2004', 'S2005'],
            'study_hours_per_day':     [6.5, 2.0, 4.0, 1.0, 8.0],
            'attendance_percentage':   [92.0, 65.0, 78.0, 55.0, 98.0],
            'mental_health_rating':    [8.0, 4.0, 6.0, 3.0, 9.0],
            'sleep_hours':             [7.5, 5.0, 6.5, 8.0, 7.0],
            'part_time_job':           ['No', 'Yes', 'No', 'Yes', 'No'],
        }
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)
        csv_bytes = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Sample CSV", data=csv_bytes,
                           file_name="gradeinsight_sample_5students.csv", mime="text/csv")

    st.divider()

    # ── File Upload ────────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)
        n_students = len(raw_df)

        st.success(f"✅ File loaded — **{n_students} records** detected")

        # ── AUTOMATIC FEATURE SELECTION ────────────────────────────────────────
        st.markdown("### 🤖 Automatic Feature Analysis")

        found_cols    = [f for f in MODEL_FEATURES if f in raw_df.columns]
        missing_cols  = [f for f in MODEL_FEATURES if f not in raw_df.columns]
        extra_cols    = [c for c in raw_df.columns if c not in MODEL_FEATURES and c != 'student_id']

        col1, col2, col3 = st.columns(3)
        col1.metric("Features Found", f"{len(found_cols)} / {len(MODEL_FEATURES)}", help="Columns from your CSV used directly")
        col2.metric("Features Auto-Filled", len(missing_cols), help="Missing columns filled with dataset averages")
        col3.metric("Extra Columns (Ignored)", len(extra_cols), help="Columns not used by the model")

        st.markdown("**Feature Detection Map:**")
        tag_html = ""
        for f in MODEL_FEATURES:
            label = FEATURE_LABELS.get(f, f)
            if f in found_cols:
                tag_html += f'<span class="feature-tag tag-found">✓ {label}</span>'
            else:
                default_val = FEATURE_DEFAULTS[f]
                tag_html += f'<span class="feature-tag tag-imputed">⚡ {label} = {default_val} (auto)</span>'
        st.markdown(tag_html, unsafe_allow_html=True)

        if missing_cols:
            st.info(f"ℹ️ **{len(missing_cols)} feature(s)** not found in your CSV. "
                    f"The model will use dataset-average defaults for these columns. "
                    f"Predictions are still fully valid.")

        # ── Encoding & Prediction ──────────────────────────────────────────────
        st.markdown("### 🚀 Running Predictions")
        progress = st.progress(0, text="Encoding features...")
        time.sleep(0.3)

        encoded_df = encode_batch_df(raw_df)
        progress.progress(40, text="Running model inference...")
        time.sleep(0.3)

        predictions = model.predict(encoded_df.values)
        predictions = np.clip(predictions, 0, 100)
        progress.progress(80, text="Generating results...")
        time.sleep(0.2)

        grades = [score_to_grade(s)[0] for s in predictions]

        # Build result dataframe
        result_df = raw_df.copy()
        if 'exam_score' in result_df.columns:
            result_df = result_df.drop(columns=['exam_score'])
        result_df['predicted_score'] = predictions.round(2)
        result_df['predicted_grade'] = grades

        progress.progress(100, text="Done!")
        time.sleep(0.2)
        progress.empty()

        # ── Results Summary ────────────────────────────────────────────────────
        st.markdown("### 📊 Prediction Results")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Students Processed", n_students)
        r2.metric("Avg Predicted Score", f"{predictions.mean():.1f}")
        r3.metric("Highest Score", f"{predictions.max():.1f}")
        r4.metric("Lowest Score", f"{predictions.min():.1f}")

        # Grade distribution chart
        grade_counts = pd.Series(grades).value_counts().reset_index()
        grade_counts.columns = ["Grade", "Count"]
        grade_order = ["A+", "A", "B", "C", "D", "F"]
        grade_counts["Grade"] = pd.Categorical(grade_counts["Grade"], categories=grade_order, ordered=True)
        grade_counts = grade_counts.sort_values("Grade")

        col_chart, col_table = st.columns([1, 1])
        with col_chart:
            fig = px.bar(grade_counts, x="Grade", y="Count",
                         title="Grade Distribution",
                         color="Grade",
                         color_discrete_map={"A+": "#06d6a0", "A": "#06d6a0",
                                             "B": "#63caff", "C": "#ffd166",
                                             "D": "#ff9f43", "F": "#ff6b6b"},
                         template="plotly_dark")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with col_table:
            st.markdown("**Score Distribution**")
            score_bins = pd.cut(predictions, bins=[0, 50, 60, 70, 80, 90, 100],
                                labels=["<50 (F)", "50-60 (D)", "60-70 (C)", "70-80 (B)", "80-90 (A)", "90-100 (A+)"])
            dist = score_bins.value_counts().sort_index().reset_index()
            dist.columns = ["Range", "Count"]
            dist["Percentage"] = (dist["Count"] / n_students * 100).round(1).astype(str) + "%"
            st.dataframe(dist, use_container_width=True, hide_index=True)

        # Full results table
        st.markdown("**All Predictions**")
        st.dataframe(result_df, use_container_width=True)

        # ── Download Button ────────────────────────────────────────────────────
        csv_out = result_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Full Predictions CSV",
            data=csv_out,
            file_name="gradeinsight_predictions.csv",
            mime="text/csv",
            type="primary"
        )

        # ── Feature Summary used ───────────────────────────────────────────────
        with st.expander("🔎 Show Encoded Features Used for Prediction"):
            st.dataframe(encoded_df.head(10), use_container_width=True)

    else:
        st.markdown("""
        <div style="border:1px dashed rgba(99,202,255,0.2);border-radius:16px;
                    padding:3rem;text-align:center;margin-top:1rem;">
            <div style="font-size:3rem;margin-bottom:1rem;opacity:0.5;">📂</div>
            <div style="color:#6b7c93;font-family:monospace;font-size:0.85rem;">
                Upload a CSV to start bulk predictions.<br>
                Download the sample template above if you need a starting point.
            </div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Model Performance":
    st.subheader("Model Performance")
    df1 = joblib.load("results_model.pkl")
    table1, = st.tabs(["Model Comparison"])
    with table1:
        st.dataframe(df1, use_container_width=True)

    fig3 = px.bar(df1, x="model", y="r2",
                  title="Model Comparison",
                  labels={"model": "Models", "r2": "R² Score"})
    st.plotly_chart(fig3, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2.5rem 0 1rem;color:#6b7c93;
            font-family:'Space Mono',monospace;font-size:0.65rem;letter-spacing:0.12em;">
    Created By Sahal Fitter · GradeInsight v2.0
</div>
""", unsafe_allow_html=True)
