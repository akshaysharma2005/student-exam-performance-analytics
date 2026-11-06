import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.data_loader import load_student_data, train_score_predictors, EDU_ORDER
from src.visualizer import (
    build_2d_correlations,
    build_equity_heatmap,
    build_3d_triaxial_space,
    build_4d_education_ladder_animation
)

st.set_page_config(
    page_title="Student Academic Performance & 4D Equity Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    
    .academic-card {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.95));
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.25);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_data():
    return load_student_data("data/students_performance.csv")

df = get_data()

@st.cache_resource
def get_models():
    return train_score_predictors(df)

models = get_models()

# Sidebar
st.sidebar.title("🎓 Academic Telemetry")
st.sidebar.markdown("Tri-Axial 3D Score Geometries, Equity Manifolds & Predictive Intervention AI.")

gender_filter = st.sidebar.multiselect("Gender", options=df["gender"].unique(), default=df["gender"].unique())
ethnicity_filter = st.sidebar.multiselect("Race / Ethnicity", options=sorted(df["race/ethnicity"].unique()), default=sorted(df["race/ethnicity"].unique()))
prep_filter = st.sidebar.multiselect("Test Preparation Course", options=df["test preparation course"].unique(), default=df["test preparation course"].unique())
lunch_filter = st.sidebar.multiselect("Lunch Type", options=df["lunch"].unique(), default=df["lunch"].unique())

filtered_df = df[
    (df["gender"].isin(gender_filter)) &
    (df["race/ethnicity"].isin(ethnicity_filter)) &
    (df["test preparation course"].isin(prep_filter)) &
    (df["lunch"].isin(lunch_filter))
]

# Header KPIs
col_title, col_kpis = st.columns([2, 3])
with col_title:
    st.title("Student Performance & 4D Equity Analyzer")
    st.caption("3D tri-axial score spaces, socioeconomic factor equity analysis, and predictive academic intervention simulation.")

total_cohort = len(filtered_df)
avg_math = filtered_df["math score"].mean() if total_cohort > 0 else 0
avg_reading = filtered_df["reading score"].mean() if total_cohort > 0 else 0
avg_writing = filtered_df["writing score"].mean() if total_cohort > 0 else 0
pass_rate = (filtered_df["passed_all"].sum() / max(1, total_cohort)) * 100

with col_kpis:
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"""
        <div class="academic-card">
            <span style="color: #94a3b8; font-size: 11px;">COHORT SIZE</span>
            <h3 style="margin: 0; color: #38bdf8;">{total_cohort:,}</h3>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="academic-card">
            <span style="color: #94a3b8; font-size: 11px;">PASS ALL 3 SUBJECTS</span>
            <h3 style="margin: 0; color: #10b981;">{pass_rate:.1f}%</h3>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="academic-card">
            <span style="color: #94a3b8; font-size: 11px;">AVG MATH SCORE</span>
            <h3 style="margin: 0; color: #f59e0b;">{avg_math:.1f} / 100</h3>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="academic-card">
            <span style="color: #94a3b8; font-size: 11px;">AVG LITERACY (R+W)</span>
            <h3 style="margin: 0; color: #a855f7;">{((avg_reading + avg_writing) / 2):.1f} / 100</h3>
        </div>
        """, unsafe_allow_html=True)

# Tabs
tab_2d, tab_3d, tab_4d, tab_ml = st.tabs([
    "📊 2D Equity & Score Correlations",
    "🌐 3D Tri-Axial Academic Space",
    "⏳ 4D Parental Education Ladder Shift",
    "🤖 Predictive Intervention Simulator"
])

# TAB 1: 2D
with tab_2d:
    col_c1, col_c2 = st.columns([3, 2])
    with col_c1:
        st.plotly_chart(build_2d_correlations(filtered_df), use_container_width=True)
    with col_c2:
        st.plotly_chart(build_equity_heatmap(filtered_df), use_container_width=True)
        
    st.divider()
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown("#### Score Distribution by Gender")
        gender_fig = px.violin(filtered_df, x="gender", y="average_score", color="gender", box=True, points="all",
                               color_discrete_map={"female": "#f43f5e", "male": "#38bdf8"})
        gender_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
        st.plotly_chart(gender_fig, use_container_width=True)
    with col_d2:
        st.markdown("#### Grade Categorization Breakdown")
        grade_counts = filtered_df["grade"].value_counts().reset_index()
        grade_counts.columns = ["Grade", "Students"]
        grade_pie = px.pie(grade_counts, names="Grade", values="Students", hole=0.45,
                           color_discrete_sequence=["#10b981", "#38bdf8", "#f59e0b", "#ec4899", "#f43f5e"])
        grade_pie.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#f8fafc"))
        st.plotly_chart(grade_pie, use_container_width=True)

# TAB 2: 3D
with tab_3d:
    st.subheader("3D Tri-Axial Score Space: Math × Reading × Writing")
    st.markdown("Students positioned in 3D coordinate space. The dashed diagonal marks balanced performance across all subjects. The 3D green mesh highlights the Convex Hull envelope of students who completed test preparation.")
    
    col_t1, col_t2 = st.columns([3, 1])
    with col_t2:
        color_mode = st.selectbox("Color Mapping", ["grade", "test preparation course", "lunch", "gender"], index=0)
        show_hull_toggle = st.checkbox("Show Test Prep Convex Hull", value=True)
        
    tri_fig = build_3d_triaxial_space(filtered_df, color_by=color_mode, show_hull=show_hull_toggle)
    st.plotly_chart(tri_fig, use_container_width=True)

# TAB 3: 4D
with tab_4d:
    st.subheader("4D Parental Education Ladder Shift (Spatio-Temporal Evolution)")
    st.markdown("Click **Play** to observe how cohort score geometry shifts across the educational ladder from *Some High School* up through *Master's Degree*.")
    st.plotly_chart(build_4d_education_ladder_animation(filtered_df), use_container_width=True)

# TAB 4: ML
with tab_ml:
    st.subheader("🤖 Predictive Academic Score & Intervention Simulator")
    st.markdown("Forecast expected student examination marks based on demographic and preparation factors.")
    
    col_i1, col_i2, col_i3 = st.columns(3)
    with col_i1:
        in_gender = st.selectbox("Student Gender", ["female", "male"])
        in_ethnicity = st.selectbox("Race / Ethnicity Group", sorted(df["race/ethnicity"].unique()))
    with col_i2:
        in_edu = st.selectbox("Parental Education Level", EDU_ORDER)
        in_lunch = st.selectbox("Lunch Support", ["standard", "free/reduced"])
    with col_i3:
        in_prep = st.selectbox("Test Preparation Course", ["none", "completed"])
        
    sim_input = pd.DataFrame([{
        "gender": in_gender,
        "race/ethnicity": in_ethnicity,
        "parental level of education": in_edu,
        "lunch": in_lunch,
        "test preparation course": in_prep
    }])
    
    pred_math = models["math score"].predict(sim_input)[0]
    pred_read = models["reading score"].predict(sim_input)[0]
    pred_write = models["writing score"].predict(sim_input)[0]
    pred_avg = (pred_math + pred_read + pred_write) / 3.0
    
    st.markdown("---")
    st.markdown("### Projected Examination Outcomes")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(f"<div class='academic-card'><span style='color: #94a3b8;'>PREDICTED MATH</span><h2 style='margin: 0; color: #f59e0b;'>{pred_math:.1f}</h2></div>", unsafe_allow_html=True)
    with r2:
        st.markdown(f"<div class='academic-card'><span style='color: #94a3b8;'>PREDICTED READING</span><h2 style='margin: 0; color: #38bdf8;'>{pred_read:.1f}</h2></div>", unsafe_allow_html=True)
    with r3:
        st.markdown(f"<div class='academic-card'><span style='color: #94a3b8;'>PREDICTED WRITING</span><h2 style='margin: 0; color: #10b981;'>{pred_write:.1f}</h2></div>", unsafe_allow_html=True)
    with r4:
        st.markdown(f"<div class='academic-card'><span style='color: #94a3b8;'>OVERALL PREDICTED AVERAGE</span><h2 style='margin: 0; color: #a855f7;'>{pred_avg:.1f}</h2></div>", unsafe_allow_html=True)
        
    st.markdown("#### Uplift Analysis")
    if in_prep == "none":
        sim_prep_boost = sim_input.copy()
        sim_prep_boost["test preparation course"] = "completed"
        boost_math = models["math score"].predict(sim_prep_boost)[0] - pred_math
        boost_read = models["reading score"].predict(sim_prep_boost)[0] - pred_read
        boost_write = models["writing score"].predict(sim_prep_boost)[0] - pred_write
        st.info(f"💡 **Intervention Opportunity**: Enrolling this student in a Test Preparation Course is projected to add **+{boost_math:.1f} Math**, **+{boost_read:.1f} Reading**, and **+{boost_write:.1f} Writing** points!")
