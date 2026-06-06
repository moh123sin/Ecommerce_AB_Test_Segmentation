import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="E-Commerce A/B Test & Segmentation", layout="wide", page_icon="🛒")

st.markdown("""
<style>
h1 {color:#2E74B5;}
.sig   {background:#e0ffe0;padding:15px;border-radius:8px;border-left:5px solid #27ae60;}
.nosig {background:#fff3e0;padding:15px;border-radius:8px;border-left:5px solid #f39c12;}
.card  {background:#f8f9fa;padding:12px;border-radius:8px;border-left:4px solid #2E74B5;margin:4px 0;}
</style>""", unsafe_allow_html=True)

# ── Data generation ───────────────────────────────────────────────────────────
@st.cache_data
def generate_ab_data(n=5000):
    np.random.seed(42)
    group = np.random.choice(["Control","Treatment"], n)
    # Treatment (new checkout) has higher conversion
    conv_rate = np.where(group == "Control", 0.12, 0.155)
    converted  = np.random.binomial(1, conv_rate, n)
    # Revenue conditional on conversion
    revenue = np.where(
        converted == 1,
        np.where(group == "Control",
                 np.random.normal(48, 15, n),
                 np.random.normal(54, 18, n)),
        0
    )
    revenue = np.clip(revenue, 0, 300)
    session_duration = np.random.normal(180, 60, n) + np.where(group == "Treatment", 20, 0)
    pages_viewed     = np.random.randint(1, 12, n)
    device           = np.random.choice(["Mobile","Desktop","Tablet"], n, p=[0.55,0.35,0.10])
    age              = np.random.randint(18, 70, n)
    return pd.DataFrame({
        "user_id": range(1, n+1),
        "group": group, "converted": converted,
        "revenue": revenue.round(2),
        "session_duration": session_duration.round(0).astype(int),
        "pages_viewed": pages_viewed, "device": device, "age": age
    })

@st.cache_data
def generate_rfm_data(n=2000):
    np.random.seed(99)
    recency   = np.random.exponential(60, n).astype(int) + 1
    frequency = np.random.poisson(5, n) + 1
    monetary  = np.random.lognormal(4, 1, n).round(2)
    return pd.DataFrame({"recency": recency, "frequency": frequency, "monetary": monetary})

df   = generate_ab_data()
rfm  = generate_rfm_data()

st.title("🛒 E-Commerce A/B Test & Customer Segmentation")
st.markdown("*Statistical testing on checkout redesign + RFM customer segmentation*")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(["🧪 A/B Test Results", "📊 Segment Deep Dive", "👥 RFM Segmentation", "📋 Raw Data"])

# ════════════════════════════════════════════════════════
# TAB 1 — A/B Test Results
# ════════════════════════════════════════════════════════
with tab1:
    st.subheader("🧪 Checkout Redesign A/B Test")
    st.markdown("**Hypothesis:** The new checkout flow (Treatment) increases conversion rate vs the original (Control).")

    ctrl  = df[df["group"] == "Control"]
    treat = df[df["group"] == "Treatment"]

    # Stats
    ctrl_conv  = ctrl["converted"].mean()
    treat_conv = treat["converted"].mean()
    lift       = (treat_conv - ctrl_conv) / ctrl_conv * 100
    ctrl_rev   = ctrl["revenue"].mean()
    treat_rev  = treat["revenue"].mean()

    # Z-test for proportions
    n_ctrl, n_treat = len(ctrl), len(treat)
    x_ctrl  = ctrl["converted"].sum()
    x_treat = treat["converted"].sum()
    p_pool  = (x_ctrl + x_treat) / (n_ctrl + n_treat)
    se      = np.sqrt(p_pool * (1 - p_pool) * (1/n_ctrl + 1/n_treat))
    z_stat  = (treat_conv - ctrl_conv) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    ci_low  = (treat_conv - ctrl_conv) - 1.96 * se
    ci_high = (treat_conv - ctrl_conv) + 1.96 * se

    # Revenue t-test
    t_stat, p_rev = stats.ttest_ind(
        treat[treat["converted"]==1]["revenue"],
        ctrl[ctrl["converted"]==1]["revenue"]
    )

    # KPIs
    k1,k2,k3,k4 = st.columns(4)
    k1.metric("Control Conversion",   f"{ctrl_conv*100:.2f}%")
    k2.metric("Treatment Conversion", f"{treat_conv*100:.2f}%", f"{lift:+.1f}% lift")
    k3.metric("p-value",              f"{p_value:.4f}")
    k4.metric("Statistically Sig?",   "✅ Yes" if p_value < 0.05 else "❌ No")

    # Result banner
    if p_value < 0.05:
        st.markdown(f"""<div class='sig'>
        <h4>✅ Statistically Significant Result (p = {p_value:.4f} < 0.05)</h4>
        <p>The Treatment group shows a <strong>{lift:.1f}% lift</strong> in conversion rate
        ({ctrl_conv*100:.2f}% → {treat_conv*100:.2f}%).
        95% CI for difference: [{ci_low*100:.2f}%, {ci_high*100:.2f}%].
        <strong>Recommendation: Ship the new checkout flow.</strong></p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class='nosig'>
        <h4>⚠️ Not Statistically Significant (p = {p_value:.4f})</h4>
        <p>Insufficient evidence to conclude the treatment outperforms control. Collect more data.</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Conversion Rate Comparison**")
        fig, ax = plt.subplots(figsize=(5,3.5))
        bars = ax.bar(["Control","Treatment"],
                      [ctrl_conv*100, treat_conv*100],
                      color=["#5BA3E0","#27ae60"], width=0.4)
        ax.set_ylabel("Conversion Rate (%)")
        ax.set_title("A/B Test: Conversion Rate")
        ax.set_ylim(0, max(ctrl_conv, treat_conv)*100 * 1.3)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.2,
                    f'{b.get_height():.2f}%', ha='center', fontsize=10, fontweight='bold')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Revenue per Converted User**")
        fig, ax = plt.subplots(figsize=(5,3.5))
        ctrl_rev_data  = ctrl[ctrl["converted"]==1]["revenue"]
        treat_rev_data = treat[treat["converted"]==1]["revenue"]
        ax.hist(ctrl_rev_data,  bins=30, alpha=0.6, color="#5BA3E0", label=f"Control (μ=${ctrl_rev_data.mean():.1f})")
        ax.hist(treat_rev_data, bins=30, alpha=0.6, color="#27ae60",  label=f"Treatment (μ=${treat_rev_data.mean():.1f})")
        ax.axvline(ctrl_rev_data.mean(),  color="#2E74B5", linestyle="--", lw=1.5)
        ax.axvline(treat_rev_data.mean(), color="#1a7a40", linestyle="--", lw=1.5)
        ax.set_xlabel("Revenue ($)"); ax.set_ylabel("Count")
        ax.set_title(f"Revenue Distribution (p={p_rev:.3f})")
        ax.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Breakdown by device
    st.markdown("**Conversion Rate by Device**")
    device_conv = df.groupby(["device","group"])["converted"].mean().unstack() * 100
    fig, ax = plt.subplots(figsize=(8,3))
    x = np.arange(len(device_conv))
    w = 0.35
    ax.bar(x - w/2, device_conv["Control"],   w, label="Control",   color="#5BA3E0")
    ax.bar(x + w/2, device_conv["Treatment"], w, label="Treatment", color="#27ae60")
    ax.set_xticks(x); ax.set_xticklabels(device_conv.index)
    ax.set_ylabel("Conversion Rate (%)")
    ax.set_title("Conversion by Device Type")
    ax.legend()
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # Stats summary table
    st.markdown("**Statistical Summary**")
    summary = pd.DataFrame({
        "Metric": ["Sample Size","Conversions","Conversion Rate","Avg Revenue/User","Avg Order Value"],
        "Control": [n_ctrl, x_ctrl, f"{ctrl_conv*100:.2f}%",
                    f"${ctrl['revenue'].mean():.2f}", f"${ctrl_rev_data.mean():.2f}"],
        "Treatment": [n_treat, x_treat, f"{treat_conv*100:.2f}%",
                      f"${treat['revenue'].mean():.2f}", f"${treat_rev_data.mean():.2f}"]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════
# TAB 2 — Segment Deep Dive
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("📊 Segment Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Conversion by Age Group**")
        df2 = df.copy()
        df2["age_group"] = pd.cut(df2["age"], bins=[17,25,35,50,70],
                                   labels=["18-25","26-35","36-50","51+"])
        age_conv = df2.groupby(["age_group","group"])["converted"].mean().unstack() * 100
        fig, ax = plt.subplots(figsize=(5,3.5))
        x = np.arange(len(age_conv)); w = 0.35
        ax.bar(x-w/2, age_conv.get("Control",   pd.Series([0]*len(age_conv))), w, label="Control",   color="#5BA3E0")
        ax.bar(x+w/2, age_conv.get("Treatment", pd.Series([0]*len(age_conv))), w, label="Treatment", color="#27ae60")
        ax.set_xticks(x); ax.set_xticklabels(age_conv.index)
        ax.set_ylabel("Conversion Rate (%)"); ax.set_title("Conversion by Age Group"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Session Duration vs Conversion**")
        fig, ax = plt.subplots(figsize=(5,3.5))
        ax.hist(df[df["converted"]==0]["session_duration"], bins=30, alpha=0.6,
                color="#e74c3c", label="Not Converted", density=True)
        ax.hist(df[df["converted"]==1]["session_duration"], bins=30, alpha=0.6,
                color="#27ae60", label="Converted", density=True)
        ax.set_xlabel("Session Duration (sec)"); ax.set_ylabel("Density")
        ax.set_title("Session Duration by Conversion"); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("**Pages Viewed vs Conversion Rate**")
    pv = df.groupby("pages_viewed")["converted"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10,3))
    ax.bar(pv["pages_viewed"], pv["converted"]*100, color="#2E74B5")
    ax.set_xlabel("Pages Viewed"); ax.set_ylabel("Conversion Rate (%)")
    ax.set_title("More Page Views = Higher Conversion?")
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════
# TAB 3 — RFM Segmentation
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("👥 RFM Customer Segmentation")
    st.markdown("Segment customers by **Recency**, **Frequency**, and **Monetary** value using K-Means clustering.")

    n_clusters = st.slider("Number of Segments", 3, 6, 4)

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Segment"] = km.fit_predict(rfm_scaled)

    # Label segments by monetary value
    seg_means = rfm.groupby("Segment")["monetary"].mean().sort_values(ascending=False)
    labels_map = {seg: name for seg, name in zip(
        seg_means.index,
        ["Champions","Loyal Customers","At-Risk","Lost Customers",
         "Potential Loyalists","New Customers"][:n_clusters]
    )}
    rfm["Segment_Name"] = rfm["Segment"].map(labels_map)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Segment Distribution**")
        seg_counts = rfm["Segment_Name"].value_counts()
        colors = ["#2E74B5","#27ae60","#e67e22","#e74c3c","#8e44ad","#1abc9c"][:n_clusters]
        fig, ax = plt.subplots(figsize=(5,4))
        ax.pie(seg_counts.values, labels=seg_counts.index, autopct="%1.1f%%",
               colors=colors, startangle=90)
        ax.set_title("Customer Segment Distribution")
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with col2:
        st.markdown("**Average RFM by Segment**")
        seg_profile = rfm.groupby("Segment_Name")[["recency","frequency","monetary"]].mean().round(1)
        seg_profile.columns = ["Avg Recency (days)","Avg Frequency","Avg Monetary ($)"]
        st.dataframe(seg_profile.sort_values("Avg Monetary ($)", ascending=False),
                     use_container_width=True)

    st.markdown("**RFM Scatter: Frequency vs Monetary (coloured by Segment)**")
    fig, ax = plt.subplots(figsize=(10,4))
    for i, (seg, grp) in enumerate(rfm.groupby("Segment_Name")):
        ax.scatter(grp["frequency"], grp["monetary"], label=seg,
                   alpha=0.4, s=15, color=colors[i % len(colors)])
    ax.set_xlabel("Frequency (orders)"); ax.set_ylabel("Monetary Value ($)")
    ax.set_title("Customer Segments: Frequency vs Monetary Value")
    ax.legend(fontsize=8, markerscale=2)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("**Actionable Recommendations by Segment**")
    recs = {
        "Champions":          "Reward them. Upsell premium products. Ask for reviews.",
        "Loyal Customers":    "Offer loyalty programmes. Upsell higher-value items.",
        "At-Risk":            "Send win-back campaigns. Offer discounts. Ask for feedback.",
        "Lost Customers":     "Re-engage with aggressive offers or accept churn.",
        "Potential Loyalists":"Offer membership / loyalty programmes to convert them.",
        "New Customers":      "Onboarding emails, first-purchase discounts, tutorials.",
    }
    for seg in seg_counts.index:
        if seg in recs:
            st.markdown(f"<div class='card'><strong>{seg}</strong>: {recs[seg]}</div>",
                        unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 4 — Raw Data
# ════════════════════════════════════════════════════════
with tab4:
    st.subheader("📋 A/B Test Raw Data")
    col1, col2 = st.columns(2)
    with col1:
        group_filter = st.multiselect("Group", ["Control","Treatment"], default=["Control","Treatment"])
    with col2:
        conv_filter = st.multiselect("Converted", [0,1], default=[0,1])
    filtered = df[(df["group"].isin(group_filter)) & (df["converted"].isin(conv_filter))]
    st.dataframe(filtered.head(200), use_container_width=True)
    st.download_button("📥 Download CSV", data=df.to_csv(index=False),
                       file_name="ab_test_data.csv", mime="text/csv")
