import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import anthropic
import math

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Credit Risk Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Serif Display', serif;
    background-color: #060d18;
    color: #c8d8f0;
}

/* Main background */
.stApp { background-color: #060d18; }
section[data-testid="stSidebar"] { background-color: #0a1626 !important; border-right: 1px solid #0f1923; }
section[data-testid="stSidebar"] * { color: #c8d8f0 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background-color: #0a1626;
    border: 1px solid #0f1923;
    border-radius: 12px;
    padding: 16px !important;
}
[data-testid="stMetricValue"] { font-family: 'DM Mono', monospace !important; font-size: 24px !important; }
[data-testid="stMetricLabel"] { font-family: 'DM Mono', monospace !important; font-size: 10px !important; text-transform: uppercase; letter-spacing: 2px; color: #4a5f7a !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { background-color: #0a1626; border-radius: 8px; border: 1px solid #0f1923; gap: 4px; padding: 4px; }
.stTabs [data-baseweb="tab"] { background-color: transparent; color: #4a5f7a; font-family: 'DM Mono', monospace; font-size: 11px; text-transform: uppercase; letter-spacing: 1px; border-radius: 6px; }
.stTabs [aria-selected="true"] { background-color: #3d8cf722 !important; color: #3d8cf7 !important; border: 1px solid #3d8cf7 !important; }

/* Sliders */
.stSlider [data-baseweb="slider"] { padding: 0; }

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #3d8cf7, #1a6fe8) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    width: 100% !important;
    padding: 12px !important;
}
.stButton button:hover { transform: translateY(-1px); box-shadow: 0 4px 20px #3d8cf740 !important; }

/* Text input */
.stTextInput input { background-color: #0a1626 !important; border: 1px solid #1a2b40 !important; color: #c8d8f0 !important; font-family: 'DM Mono', monospace !important; border-radius: 6px !important; }

/* Dataframe */
.stDataFrame { background-color: #0a1626; border-radius: 12px; }

/* Info/success/warning boxes */
.stAlert { border-radius: 10px; font-family: 'DM Mono', monospace; }

/* Headers */
h1, h2, h3 { color: #e8f2ff !important; }

/* Divider */
hr { border-color: #0f1923; }

/* Number input */
.stNumberInput input { background-color: #0a1626 !important; border: 1px solid #1a2b40 !important; color: #c8d8f0 !important; font-family: 'DM Mono', monospace !important; }

/* Select box */
.stSelectbox select { background-color: #0a1626 !important; color: #c8d8f0 !important; }

/* card style for st.container */
.risk-card {
    background: #0a1626;
    border: 1px solid #0f1923;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
}

.grade-display {
    font-family: 'DM Serif Display', serif;
    font-size: 64px;
    font-weight: 400;
    line-height: 1;
    text-align: center;
}

.metric-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #4a5f7a;
    margin-bottom: 4px;
}

.metric-value {
    font-family: 'DM Mono', monospace;
    font-size: 28px;
    font-weight: 700;
}

.formula-row {
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# ─── ENGINE ───────────────────────────────────────────────────────────────────

WEIGHTS = {
    "annualIncome":     {"w": 0.25, "label": "Annual Income"},
    "debtToIncome":     {"w": 0.20, "label": "Debt-to-Income Ratio"},
    "creditScore":      {"w": 0.25, "label": "Credit Score"},
    "employmentYears":  {"w": 0.15, "label": "Employment Tenure"},
    "collateralRatio":  {"w": 0.15, "label": "Collateral Ratio"},
}

def score_var(key, v):
    if key == "annualIncome":
        return 100 if v>=2000000 else 85 if v>=1200000 else 65 if v>=600000 else 45 if v>=300000 else 20
    if key == "debtToIncome":
        return 100 if v<=0.10 else 85 if v<=0.20 else 60 if v<=0.35 else 35 if v<=0.50 else 10
    if key == "creditScore":
        return 100 if v>=800 else 85 if v>=740 else 65 if v>=670 else 40 if v>=580 else 15
    if key == "employmentYears":
        return 100 if v>=10 else 80 if v>=5 else 55 if v>=2 else 35 if v>=1 else 15
    if key == "collateralRatio":
        return 100 if v>=1.5 else 80 if v>=1.2 else 60 if v>=1.0 else 35 if v>=0.7 else 10
    return 50

def compute_score(inputs):
    total, bd = 0, {}
    for key, meta in WEIGHTS.items():
        s = score_var(key, inputs[key])
        bd[key] = s
        total += s * meta["w"]
    return round(total), bd

def score_to_pd(score):
    return 1 / (1 + math.exp(5.0 - 0.08 * score))

def compute_lgd(collateral_ratio):
    cr = collateral_ratio
    return 0.10 if cr>=1.5 else 0.25 if cr>=1.2 else 0.40 if cr>=1.0 else 0.60 if cr>=0.7 else 0.80

def get_risk_grade(score):
    if score >= 90: return {"grade": "AAA", "color": "#00d68f", "label": "Prime"}
    if score >= 80: return {"grade": "AA",  "color": "#00c4a0", "label": "High Grade"}
    if score >= 72: return {"grade": "A",   "color": "#1eb8a6", "label": "Upper Medium"}
    if score >= 63: return {"grade": "BBB", "color": "#f5a623", "label": "Lower Medium"}
    if score >= 54: return {"grade": "BB",  "color": "#e8793a", "label": "Speculative"}
    if score >= 44: return {"grade": "B",   "color": "#e05252", "label": "Highly Speculative"}
    if score >= 35: return {"grade": "CCC", "color": "#c0392b", "label": "Substantial Risk"}
    return              {"grade": "D",   "color": "#8b0000", "label": "Default"}

def get_recommendation(grade):
    if grade in ["AAA","AA","A","BBB"]: return {"text": "APPROVE", "color": "#00d68f", "icon": "✅"}
    if grade in ["BB","B"]:             return {"text": "REVIEW",  "color": "#f5a623", "icon": "⚠️"}
    return                                      {"text": "DECLINE", "color": "#e05252", "icon": "❌"}

def fmt_inr(n):
    return f"₹{n:,.0f}"

# ─── PORTFOLIO DATA ────────────────────────────────────────────────────────────

PORTFOLIO = [
    {"name":"Arjun Mehta",   "annualIncome":1200000,"debtToIncome":0.18,"creditScore":780,"employmentYears":8, "collateralRatio":1.4,"loanAmount":5000000},
    {"name":"Priya Sharma",  "annualIncome":550000, "debtToIncome":0.42,"creditScore":620,"employmentYears":3, "collateralRatio":0.8,"loanAmount":2000000},
    {"name":"Rohan Kapoor",  "annualIncome":950000, "debtToIncome":0.25,"creditScore":710,"employmentYears":6, "collateralRatio":1.1,"loanAmount":3500000},
    {"name":"Sneha Patel",   "annualIncome":380000, "debtToIncome":0.55,"creditScore":560,"employmentYears":1, "collateralRatio":0.5,"loanAmount":1500000},
    {"name":"Vikram Singh",  "annualIncome":2000000,"debtToIncome":0.12,"creditScore":840,"employmentYears":15,"collateralRatio":1.8,"loanAmount":8000000},
    {"name":"Kavya Nair",    "annualIncome":720000, "debtToIncome":0.31,"creditScore":695,"employmentYears":4, "collateralRatio":1.0,"loanAmount":2800000},
    {"name":"Amit Joshi",    "annualIncome":450000, "debtToIncome":0.48,"creditScore":590,"employmentYears":2, "collateralRatio":0.65,"loanAmount":1800000},
    {"name":"Divya Reddy",   "annualIncome":1600000,"debtToIncome":0.15,"creditScore":820,"employmentYears":12,"collateralRatio":1.6,"loanAmount":6500000},
]

# ─── HEADER ───────────────────────────────────────────────────────────────────

st.markdown("""
<div style="border-bottom:1px solid #0f1923;padding-bottom:20px;margin-bottom:24px;display:flex;align-items:center;gap:12px;">
  <div style="width:8px;height:40px;background:linear-gradient(180deg,#3d8cf7,#00d68f);border-radius:4px;flex-shrink:0;"></div>
  <div>
    <h1 style="font-size:26px;font-weight:400;letter-spacing:-0.5px;margin:0;">Credit Risk Intelligence</h1>
    <p style="font-family:'DM Mono',monospace;font-size:10px;color:#4a5f7a;letter-spacing:2px;text-transform:uppercase;margin:4px 0 0 0;">
      Scorecard · PD/LGD · Expected Loss · Portfolio
    </p>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── TABS ─────────────────────────────────────────────────────────────────────

tab_scorer, tab_portfolio = st.tabs(["📋  SCORER", "📊  PORTFOLIO"])

# ══════════════════════════════════════════════════════════════════════════════
# SCORER TAB
# ══════════════════════════════════════════════════════════════════════════════

with tab_scorer:
    sidebar = st.sidebar
    sidebar.markdown("""
    <div style="padding:8px 0 16px 0;border-bottom:1px solid #0f1923;margin-bottom:20px;">
      <div style="font-family:'DM Mono',monospace;font-size:10px;color:#4a5f7a;letter-spacing:2px;text-transform:uppercase;">
        Applicant Inputs
      </div>
    </div>
    """, unsafe_allow_html=True)

    # API Key input
    api_key = sidebar.text_input("🔑 Anthropic API Key", type="password", placeholder="sk-ant-api03-...")
    sidebar.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    name = sidebar.text_input("Applicant Name", value="New Applicant")
    sidebar.markdown("---")

    annual_income     = sidebar.slider("Annual Income (₹)",      100000,  10000000, 600000,  50000)
    debt_to_income    = sidebar.slider("Debt-to-Income Ratio",   0.05,   0.80,   0.28,   0.01, format="%.2f")
    credit_score      = sidebar.slider("Credit Score (CIBIL)",   300,    900,    710,    5)
    employment_years  = sidebar.slider("Employment Tenure (yrs)",0,      30,     5,      1)
    collateral_ratio  = sidebar.slider("Collateral Ratio (x)",   0.20,   2.50,   1.10,   0.05, format="%.2f")
    loan_amount       = sidebar.slider("Loan Amount / EAD (₹)",  100000, 50000000, 2000000, 100000)

    sidebar.markdown("---")
    gen_commentary = sidebar.button("⚡ Generate AI Commentary")

    # ── COMPUTE ──
    inputs = {
        "annualIncome":    annual_income,
        "debtToIncome":    debt_to_income,
        "creditScore":     credit_score,
        "employmentYears": employment_years,
        "collateralRatio": collateral_ratio,
        "loanAmount":      loan_amount,
    }

    total_score, breakdown = compute_score(inputs)
    pd_val   = score_to_pd(total_score)
    lgd_val  = compute_lgd(collateral_ratio)
    ead_val  = loan_amount
    el_val   = pd_val * lgd_val * ead_val
    rg       = get_risk_grade(total_score)
    rec      = get_recommendation(rg["grade"])

    # ── TOP ROW: Score · Grade · Decision ──
    col1, col2, col3 = st.columns(3)

    with col1:
        # Gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=total_score,
            domain={"x":[0,1],"y":[0,1]},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#2a3a50","tickfont":{"color":"#4a5f7a","family":"DM Mono"}},
                "bar":{"color":rg["color"],"thickness":0.25},
                "bgcolor":"#0a1626",
                "borderwidth":0,
                "steps":[
                    {"range":[0,44],  "color":"#0f1923"},
                    {"range":[44,63], "color":"#0d1a28"},
                    {"range":[63,80], "color":"#0a1626"},
                    {"range":[80,100],"color":"#081420"},
                ],
                "threshold":{"line":{"color":rg["color"],"width":2},"thickness":0.7,"value":total_score}
            },
            number={"font":{"color":rg["color"],"family":"DM Mono","size":36},"suffix":"/100"}
        ))
        fig_gauge.update_layout(
            height=200, margin=dict(l=20,r=20,t=20,b=0),
            paper_bgcolor="#0a1626", font_color="#c8d8f0",
        )
        st.markdown('<div class="risk-card"><div class="metric-label">Composite Score</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="risk-card" style="text-align:center;min-height:220px;display:flex;flex-direction:column;justify-content:center;align-items:center;">
          <div class="metric-label" style="margin-bottom:12px;">Risk Grade</div>
          <div class="grade-display" style="color:{rg['color']};">{rg['grade']}</div>
          <div style="font-family:'DM Mono',monospace;font-size:12px;color:{rg['color']}88;margin-top:8px;">{rg['label']}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="risk-card" style="text-align:center;min-height:220px;display:flex;flex-direction:column;justify-content:center;align-items:center;">
          <div class="metric-label" style="margin-bottom:12px;">Credit Decision</div>
          <div style="font-size:48px;margin-bottom:8px;">{rec['icon']}</div>
          <div style="font-family:'DM Mono',monospace;font-size:22px;font-weight:700;color:{rec['color']};">{rec['text']}</div>
          <div style="font-family:'DM Mono',monospace;font-size:11px;color:#4a5f7a;margin-top:8px;">{name}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── METRIC CARDS: PD / LGD / EAD / EL ──
    m1, m2, m3, m4 = st.columns(4)

    pd_color  = "#e05252" if pd_val>0.15 else "#f5a623" if pd_val>0.05 else "#00d68f"
    el_color  = "#e05252" if el_val>100000 else "#f5a623" if el_val>30000 else "#00d68f"

    with m1:
        st.markdown(f"""
        <div class="risk-card">
          <div class="metric-label">Probability of Default</div>
          <div class="metric-value" style="color:{pd_color};">{pd_val*100:.2f}%</div>
          <div style="font-family:'DM Mono',monospace;font-size:9px;color:#2a3a50;margin-top:4px;">PD — Logistic Regression</div>
        </div>""", unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
        <div class="risk-card">
          <div class="metric-label">Loss Given Default</div>
          <div class="metric-value" style="color:#3d8cf7;">{lgd_val*100:.0f}%</div>
          <div style="font-family:'DM Mono',monospace;font-size:9px;color:#2a3a50;margin-top:4px;">LGD — Collateral Recovery</div>
        </div>""", unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
        <div class="risk-card">
          <div class="metric-label">Exposure at Default</div>
          <div class="metric-value" style="color:#8899b4;">{fmt_inr(ead_val)}</div>
          <div style="font-family:'DM Mono',monospace;font-size:9px;color:#2a3a50;margin-top:4px;">EAD — Loan Amount</div>
        </div>""", unsafe_allow_html=True)

    with m4:
        st.markdown(f"""
        <div class="risk-card">
          <div class="metric-label">Expected Loss</div>
          <div class="metric-value" style="color:{el_color};">{fmt_inr(el_val)}</div>
          <div style="font-family:'DM Mono',monospace;font-size:9px;color:#2a3a50;margin-top:4px;">EL = PD × LGD × EAD</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── SCORECARD BREAKDOWN ──
    col_bars, col_commentary = st.columns([1, 1])

    with col_bars:
        st.markdown('<div class="risk-card"><div class="metric-label" style="margin-bottom:16px;">Scorecard Variable Breakdown</div>', unsafe_allow_html=True)

        bar_data = []
        for key, meta in WEIGHTS.items():
            s = breakdown[key]
            bar_data.append({
                "Variable": meta["label"],
                "Score": s,
                "Weight": f"w={int(meta['w']*100)}%",
                "Color": "#00d68f" if s>=80 else "#f5a623" if s>=55 else "#e05252"
            })

        fig_bars = go.Figure()
        for row in bar_data:
            fig_bars.add_trace(go.Bar(
                y=[row["Variable"]],
                x=[row["Score"]],
                orientation="h",
                marker_color=row["Color"],
                text=f"{row['Score']} pts · {row['Weight']}",
                textposition="inside",
                textfont={"family":"DM Mono","size":11,"color":"#060d18"},
                showlegend=False,
            ))

        fig_bars.update_layout(
            height=220, margin=dict(l=0,r=0,t=0,b=0),
            paper_bgcolor="#060d18", plot_bgcolor="#060d18",
            xaxis=dict(range=[0,100], showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, tickfont={"family":"DM Mono","size":11,"color":"#8899b4"}),
            barmode="overlay", bargap=0.35,
        )
        st.plotly_chart(fig_bars, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

    with col_commentary:
        st.markdown('<div class="risk-card" style="min-height:300px;"><div class="metric-label" style="margin-bottom:14px;">AI Credit Analyst Commentary</div>', unsafe_allow_html=True)

        if gen_commentary:
            if not api_key:
                st.warning("Please enter your Anthropic API Key in the sidebar.")
            else:
                with st.spinner("Analysing credit profile…"):
                    try:
                        client = anthropic.Anthropic(api_key=api_key)
                        prompt = f"""You are a senior credit analyst at a bank. Analyze this loan application and give a concise 3-4 sentence credit decision commentary.

Applicant: {name}
- Annual Income: ₹{annual_income:,.0f}
- Debt-to-Income Ratio: {debt_to_income*100:.1f}%
- Credit Score: {credit_score}
- Employment Tenure: {employment_years} years
- Collateral Ratio: {collateral_ratio:.2f}x
- Loan Amount (EAD): ₹{loan_amount:,.0f}

Risk Metrics:
- Composite Score: {total_score}/100
- Risk Grade: {rg['grade']} ({rg['label']})
- PD: {pd_val*100:.2f}%
- LGD: {lgd_val*100:.0f}%
- Expected Loss: ₹{el_val:,.0f}
- Decision: {rec['text']}

Write 3-4 sentences like a real credit analyst memo. Be specific about key risk drivers and mitigants. Use professional banking language."""

                        message = client.messages.create(
                            model="claude-sonnet-4-20250514",
                            max_tokens=1000,
                            messages=[{"role":"user","content":prompt}]
                        )
                        commentary_text = message.content[0].text
                        st.session_state["commentary"] = commentary_text
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        if "commentary" in st.session_state:
            st.markdown(f"""
            <p style="font-size:13px;line-height:1.8;color:#a0b4cc;font-style:italic;font-family:'DM Serif Display',serif;">
              {st.session_state['commentary']}
            </p>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <p style="font-size:12px;color:#2a3a50;font-family:'DM Mono',monospace;">
              Enter your API key and click '⚡ Generate AI Commentary' in the sidebar to get a senior analyst's view.
            </p>""", unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PORTFOLIO TAB
# ══════════════════════════════════════════════════════════════════════════════

with tab_portfolio:

    # Build portfolio metrics
    port_rows = []
    total_el, total_ead, total_pd = 0, 0, 0
    grade_counts = {g:0 for g in ["AAA","AA","A","BBB","BB","B","CCC","D"]}

    for b in PORTFOLIO:
        score, _ = compute_score(b)
        pd_b  = score_to_pd(score)
        lgd_b = compute_lgd(b["collateralRatio"])
        el_b  = pd_b * lgd_b * b["loanAmount"]
        rg_b  = get_risk_grade(score)
        rec_b = get_recommendation(rg_b["grade"])
        total_el  += el_b
        total_ead += b["loanAmount"]
        total_pd  += pd_b
        grade_counts[rg_b["grade"]] += 1
        port_rows.append({
            "Borrower":   b["name"],
            "Score":      score,
            "Grade":      rg_b["grade"],
            "PD (%)":     round(pd_b*100, 2),
            "LGD (%)":    int(lgd_b*100),
            "EAD (₹)":    b["loanAmount"],
            "Exp. Loss":  round(el_b),
            "Decision":   rec_b["icon"]+" "+rec_b["text"],
            "_grade_color": rg_b["color"],
        })

    avg_pd = total_pd / len(PORTFOLIO)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f'<div class="risk-card"><div class="metric-label">Total Borrowers</div><div class="metric-value" style="color:#3d8cf7;">{len(PORTFOLIO)}</div><div style="font-family:\'DM Mono\',monospace;font-size:9px;color:#2a3a50;margin-top:4px;">Portfolio Size</div></div>', unsafe_allow_html=True)
    with k2:
        apd_color = "#e05252" if avg_pd>0.1 else "#f5a623"
        st.markdown(f'<div class="risk-card"><div class="metric-label">Avg Probability of Default</div><div class="metric-value" style="color:{apd_color};">{avg_pd*100:.2f}%</div><div style="font-family:\'DM Mono\',monospace;font-size:9px;color:#2a3a50;margin-top:4px;">Portfolio Avg PD</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="risk-card"><div class="metric-label">Total Expected Loss</div><div class="metric-value" style="color:#e05252;">{fmt_inr(total_el)}</div><div style="font-family:\'DM Mono\',monospace;font-size:9px;color:#2a3a50;margin-top:4px;">EL = Σ(PD×LGD×EAD)</div></div>', unsafe_allow_html=True)
    with k4:
        st.markdown(f'<div class="risk-card"><div class="metric-label">Total Exposure (EAD)</div><div class="metric-value" style="color:#8899b4;">{fmt_inr(total_ead)}</div><div style="font-family:\'DM Mono\',monospace;font-size:9px;color:#2a3a50;margin-top:4px;">Sum of Loan Amounts</div></div>', unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # Table + Chart
    col_table, col_chart = st.columns([2, 1])

    with col_table:
        st.markdown('<div class="risk-card"><div class="metric-label" style="margin-bottom:16px;">Borrower Risk Matrix</div>', unsafe_allow_html=True)

        df = pd.DataFrame(port_rows).drop(columns=["_grade_color"])

        # Color map for grades
        grade_colors = {"AAA":"#00d68f","AA":"#00c4a0","A":"#1eb8a6","BBB":"#f5a623","BB":"#e8793a","B":"#e05252","CCC":"#c0392b","D":"#8b0000"}

        def color_grade(val):
            c = grade_colors.get(val, "#c8d8f0")
            return f"color: {c}; font-weight: bold; font-family: DM Mono, monospace;"

        def color_pd(val):
            c = "#e05252" if val>15 else "#f5a623" if val>5 else "#00d68f"
            return f"color: {c}; font-family: DM Mono, monospace;"

        def color_el(val):
            c = "#e05252" if val>50000 else "#c8d8f0"
            return f"color: {c}; font-family: DM Mono, monospace;"

        styled_df = df.style\
            .applymap(color_grade, subset=["Grade"])\
            .applymap(color_pd,    subset=["PD (%)"])\
            .applymap(color_el,    subset=["Exp. Loss"])\
            .set_properties(**{
                "background-color": "#0a1626",
                "color": "#c8d8f0",
                "font-family": "DM Mono, monospace",
                "font-size": "12px",
                "border": "1px solid #0f1923",
            })\
            .set_table_styles([
                {"selector":"th","props":[("background-color","#060d18"),("color","#4a5f7a"),("font-family","DM Mono, monospace"),("font-size","10px"),("text-transform","uppercase"),("letter-spacing","1.5px"),("border","1px solid #0f1923")]},
            ])\
            .format({"EAD (₹)": lambda x: fmt_inr(x), "Exp. Loss": lambda x: fmt_inr(x)})

        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_chart:
        # Grade distribution bar chart
        grades_list = ["AAA","AA","A","BBB","BB","B","CCC","D"]
        colors_list = [grade_colors[g] for g in grades_list]
        counts_list = [grade_counts[g] for g in grades_list]

        fig_dist = go.Figure(go.Bar(
            x=grades_list,
            y=counts_list,
            marker_color=colors_list,
            marker_line_width=0,
            text=[str(c) if c>0 else "" for c in counts_list],
            textposition="outside",
            textfont={"family":"DM Mono","size":12,"color":"#c8d8f0"},
        ))
        fig_dist.update_layout(
            height=220, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="#060d18", plot_bgcolor="#060d18",
            xaxis=dict(showgrid=False, tickfont={"family":"DM Mono","size":11,"color":"#6b7fa3"}),
            yaxis=dict(showgrid=False, showticklabels=False),
            bargap=0.3,
        )

        st.markdown('<div class="risk-card"><div class="metric-label" style="margin-bottom:16px;">Grade Distribution</div>', unsafe_allow_html=True)
        st.plotly_chart(fig_dist, use_container_width=True, config={"displayModeBar":False})
        st.markdown('</div>', unsafe_allow_html=True)

        # Formula reference
        st.markdown("""
        <div class="risk-card">
          <div class="metric-label" style="margin-bottom:12px;">Formula Reference</div>
          <div class="formula-row"><span style="color:#3d8cf7;">EL =</span><span style="color:#6b7fa3;">PD × LGD × EAD</span></div>
          <div class="formula-row"><span style="color:#3d8cf7;">PD =</span><span style="color:#6b7fa3;">1 / (1 + e^(5 − 0.08s))</span></div>
          <div class="formula-row"><span style="color:#3d8cf7;">LGD =</span><span style="color:#6b7fa3;">f(Collateral Ratio)</span></div>
          <div class="formula-row"><span style="color:#3d8cf7;">Score =</span><span style="color:#6b7fa3;">Σ wᵢ × sᵢ</span></div>
        </div>
        """, unsafe_allow_html=True)
