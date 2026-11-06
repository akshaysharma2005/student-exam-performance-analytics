import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from scipy.spatial import ConvexHull

GRADE_COLORS = {
    "A (Elite)": "#10b981",
    "B (Proficient)": "#38bdf8",
    "C (Competent)": "#f59e0b",
    "D (Passing)": "#ec4899",
    "F (Needs Intervention)": "#f43f5e"
}

def build_2d_correlations(df, x_col="math score", y_col="reading score"):
    """Bivariate correlation scatter with trendline."""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="test preparation course",
        marginal_x="histogram",
        marginal_y="histogram",
        trendline="ols",
        color_discrete_map={"none": "#f43f5e", "completed": "#10b981"},
        labels={x_col: x_col.title(), y_col: y_col.title()}
    )
    fig.update_layout(
        title=f"Bivariate Correlation: {x_col.title()} vs. {y_col.title()}",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        height=450,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def build_equity_heatmap(df):
    """Parental education vs. test prep equity matrix."""
    pivot = df.pivot_table(
        index="parental level of education",
        columns="test preparation course",
        values="average_score",
        aggfunc="mean"
    ).round(1)

    fig = px.imshow(
        pivot,
        labels=dict(x="Test Preparation Course", y="Parental Education", color="Avg Score"),
        x=pivot.columns,
        y=pivot.index,
        color_continuous_scale="Tealgrn",
        text_auto=True,
        aspect="auto"
    )
    fig.update_layout(
        title="Mean Academic Performance Matrix (Out of 100)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        height=380,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

def build_3d_triaxial_space(df, color_by="grade", show_hull=True):
    """
    3D Tri-Axial Academic Score Space:
    X: Math Score
    Y: Reading Score
    Z: Writing Score
    Color: Grade or Test Prep
    """
    fig = go.Figure()

    groups = df.groupby(color_by)
    for group_name, group_df in groups:
        color = GRADE_COLORS.get(group_name, None)
        fig.add_trace(go.Scatter3d(
            x=group_df["math score"],
            y=group_df["reading score"],
            z=group_df["writing score"],
            mode="markers",
            name=str(group_name),
            marker=dict(size=5, color=color, opacity=0.8),
            customdata=np.stack((
                group_df["gender"],
                group_df["parental level of education"],
                group_df["test preparation course"],
                group_df["average_score"].round(1)
            ), axis=-1),
            hovertemplate="<b>Average: %{customdata[3]}</b><br>Math: %{x}<br>Reading: %{y}<br>Writing: %{z}<br>Parental Edu: %{customdata[1]}<br>Prep: %{customdata[2]}<extra></extra>"
        ))

    # Add 3D Equi-performance diagonal line (where Math = Reading = Writing)
    diag = np.linspace(20, 100, 20)
    fig.add_trace(go.Scatter3d(
        x=diag, y=diag, z=diag,
        mode="lines",
        line=dict(color="rgba(255, 255, 255, 0.4)", width=4, dash="dash"),
        name="Equi-Performance Line (X=Y=Z)",
        showlegend=True
    ))

    # Add 3D Convex Hull envelope for students who completed test prep
    if show_hull:
        prep_df = df[df["test preparation course"] == "completed"]
        pts = prep_df[["math score", "reading score", "writing score"]].values
        if len(pts) >= 4:
            try:
                hull = ConvexHull(pts)
                fig.add_trace(go.Mesh3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    i=hull.simplices[:, 0], j=hull.simplices[:, 1], k=hull.simplices[:, 2],
                    color="#10b981", opacity=0.15,
                    name="Test Prep Envelope (Convex Hull)",
                    showlegend=True
                ))
            except Exception:
                pass

    fig.update_layout(
        title="3D Tri-Axial Academic Score Space (Math × Reading × Writing)",
        scene=dict(
            xaxis=dict(title="Math Score", range=[0, 105], backgroundcolor="#0f172a", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Reading Score", range=[0, 105], backgroundcolor="#0f172a", gridcolor="rgba(255,255,255,0.1)"),
            zaxis=dict(title="Writing Score", range=[0, 105], backgroundcolor="#0f172a", gridcolor="rgba(255,255,255,0.1)"),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.3))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        height=620,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5)
    )
    return fig

def build_4d_education_ladder_animation(df):
    """
    4D Spatio-Temporal Animation:
    - Dim 1 (X): Math Score
    - Dim 2 (Y): Reading Score
    - Dim 3 (Z): Writing Score
    - Dim 4 (Color): Total Aggregate Score
    - Dim 5 (Size): Average Score
    - Animation Frame: Parental Level of Education (ordered ladder from some high school to master's degree)
    """
    df_anim = df.sort_values(by="parental level of education").copy()
    df_anim["parental_edu_str"] = df_anim["parental level of education"].astype(str)

    fig = px.scatter_3d(
        df_anim,
        x="math score",
        y="reading score",
        z="writing score",
        color="total_score",
        size="average_score",
        size_max=16,
        animation_frame="parental_edu_str",
        animation_group="race/ethnicity",
        hover_data={"gender": True, "lunch": True, "test preparation course": True, "grade": True},
        color_continuous_scale="Plasma",
        range_x=[15, 105],
        range_y=[15, 105],
        range_z=[15, 105],
        range_color=[60, 300],
        labels={
            "math score": "Math Score",
            "reading score": "Reading Score",
            "writing score": "Writing Score",
            "total_score": "Total Score (0-300)"
        }
    )

    fig.update_layout(
        title="4D Parental Education Ladder Shift (Watch Score Mass Migrate Toward Mastery)",
        scene=dict(
            xaxis=dict(title="Math Score", backgroundcolor="#0b132b", gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="Reading Score", backgroundcolor="#0b132b", gridcolor="rgba(255,255,255,0.1)"),
            zaxis=dict(title="Writing Score", backgroundcolor="#0b132b", gridcolor="rgba(255,255,255,0.1)"),
            camera=dict(eye=dict(x=1.7, y=1.7, z=1.2))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc"),
        height=620,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    return fig
