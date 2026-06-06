# 🛒 E-Commerce A/B Test & Customer Segmentation

End-to-end statistical analysis of a checkout redesign A/B test combined with RFM-based customer segmentation.

## 🔗 Live Demo
[Open the E-Commerce A/B Test Dashboard](https://ecommerce-ab-test-mohsin.streamlit.app/)

## 🎯 Business Problem
An e-commerce company wants to know:
1. Does the new checkout flow significantly increase conversion rate?
2. Which customer segments should we target for retention campaigns?

## 🛠️ Tech Stack
- **Python** · Pandas · NumPy
- **SciPy** — Z-test, t-test, statistical significance
- **Scikit-learn** — K-Means clustering, StandardScaler
- **Streamlit** — interactive web app
- **Matplotlib** — visualizations

## 📊 Features
- **A/B Test Dashboard** — conversion rates, revenue comparison, statistical significance
- **Segment Analysis** — breakdown by device, age, session behaviour
- **RFM Segmentation** — K-Means clustering into Champions, Loyal, At-Risk, Lost segments
- **Actionable Recommendations** — per-segment marketing strategy
- **Data Export** to CSV

## 🧠 Methodology

### A/B Testing
1. Two-sample Z-test for proportions (conversion rate)
2. Independent t-test for revenue difference
3. 95% confidence intervals for the treatment effect
4. Segmented analysis by device and age group

### RFM Segmentation
1. Compute Recency, Frequency, Monetary scores per customer
2. Standardise with StandardScaler
3. K-Means clustering (user-configurable k=3-6)
4. Label segments by average monetary value
5. Generate actionable marketing recommendations per segment

## 🚀 Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📈 Key Results
| Metric | Control | Treatment | Significance |
|--------|---------|-----------|--------------|
| Conversion Rate | 12.0% | 15.5% | ✅ p < 0.05 |
| Avg Order Value | $48 | $54 | ✅ p < 0.05 |
| Lift | — | +29.2% | — |
