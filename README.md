# 🎓 Student Academic Performance & 4D Equity Analyzer

[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

> **Tri-Axial 3D Academic Score Spaces, Socioeconomic Equity Heatmaps, and 4D Educational Ladder Evolution across 1,000 High School Students.**

---

## 🌟 Overview

The **Student Academic Performance & 4D Equity Analyzer** brings geometric multi-dimensional analysis to educational assessment. By examining standardized exam outcomes in Math, Reading, and Writing in connection with parental education levels, test preparation programs, and socioeconomic lunch status, this application empowers educators and researchers to isolate systemic achievement barriers and simulate predictive interventions.

```
                              ┌───────────────────────────────────┐
                              │     Student Examination Data      │
                              │       (1,000 Cohort Records)      │
                              └─────────────────┬─────────────────┘
                                                │
                 ┌──────────────────────────────┼──────────────────────────────┐
                 ▼                              ▼                              ▼
      ┌─────────────────────┐        ┌─────────────────────┐        ┌─────────────────────┐
      │  2D Equity & Score  │        │   3D Tri-Axial      │        │  4D Educational     │
      │  Correlations       │        │   Academic Space    │        │  Ladder Evolution   │
      └──────────┬──────────┘        └──────────┬──────────┘        └──────────┬──────────┘
                 │                              │                              │
                 ▼                              ▼                              ▼
     • Math vs Literacy Trends       • Math × Reading × Writing     • Animated Education Ladder
     • Parental Education Matrix     • 3D Equi-Performance Axis     • Real-Time ML Predictions
     • Pass/Fail Cohort Ratios       • Test Prep 3D Convex Hull     • Dynamic Uplift Estimator
```

---

## ✨ Features & Visual Dimensions

### 📊 1. 2D Equity & Bivariate Correlation Analysis
- **Marginal Bivariate Scatterplots**: Joint examination of Math vs. Reading and Writing scores with Ordinary Least Squares (OLS) trendlines and marginal histograms categorized by preparation status.
- **Socioeconomic Performance Matrix**: Interactive heatmaps mapping parental education against test preparation courses.
- **Demographic Score Disaggregation**: Multi-distribution violin and boxplot analysis highlighting gender and ethnic performance variance.

### 🌐 2. 3D Tri-Axial Academic Score Space
- **Tri-Axial Metric System**:
  - **X-Axis**: Math Score (0-100)
  - **Y-Axis**: Reading Score (0-100)
  - **Z-Axis**: Writing Score (0-100)
  - Color-mapped by letter grade (*A Elite, B Proficient, C Competent, D Passing, F Needs Intervention*).
- **Equi-Performance Vector**: Diagonal 3D reference line ($X=Y=Z$) identifying subject specialization versus balanced academic mastery.
- **3D Delaunay Convex Hull Envelope**: Geometric enclosure framing students who completed structured test preparation.

### ⏳ 3. 4D Parental Education Ladder Shift
- **4D Hyper-Dimensional Projection**:
  - **Dim 1 (X)**: Math Score
  - **Dim 2 (Y)**: Reading Score
  - **Dim 3 (Z)**: Writing Score
  - **Dim 4 (Color)**: Cumulative Score (0-300)
  - **Dim 5 (Size)**: Average Score
  - **Dim 6 (Time / Frame Animation)**: Parental Education Ladder (*Some High School → High School → Some College → Associate's → Bachelor's → Master's*)
- Watch the student point cloud visibly migrate towards the top-right mastery quadrant as parental educational attainment rises.

### 🤖 4. Predictive Academic Intervention Simulator
- Machine learning regression pipeline delivering instant subject score forecasts and automated uplift estimates from test preparation intervention.

---

## 🚀 Getting Started

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/akshaysharma2005/student-exam-performance-analytics.git
   cd student-exam-performance-analytics
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
   ```

4. Navigate to `http://localhost:8501`.

---

## 👤 Author
- **M Akshay Sharma** ([@akshaysharma2005](https://github.com/akshaysharma2005))
- GitHub: [https://github.com/akshaysharma2005](https://github.com/akshaysharma2005)
