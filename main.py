import streamlit as st
import pandas as pd
import joblib
import requests
import json
import urllib.parse
import datetime

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(page_title="ChurnGuardian: AI Retention", layout="wide")

# ==========================================
# 2. LOAD MODEL
# ==========================================
@st.cache_resource
def load_model():
    try:
        model = joblib.load('best_churn_model.pkl')
        model_columns = joblib.load('model_columns.pkl')
        return model, model_columns
    except Exception as e:
        st.error(f"❌ Model Error: {e}")
        return None, None

model, model_columns = load_model()

# ==========================================
# 3. GENERATORS (EMAIL & CALENDAR)
# ==========================================
def generate_email_link(data):
    """Creates a pre-filled 'mailto' link for retention."""
    subject = "Special Retention Offer"
    body = f"Hi,\n\nI noticed you've been with us for {data['tenure']} months. I'd like to offer you a 20% discount to stay."
    
    safe_subject = urllib.parse.quote(subject)
    safe_body = urllib.parse.quote(body)
    return f"mailto:customer@example.com?subject={safe_subject}&body={safe_body}"

def generate_calendar_link(data):
    """Creates a Google Calendar link for 'Tomorrow at 10 AM'."""
    # 1. Calculate time: Tomorrow at 10:00 AM
    tomorrow = datetime.datetime.now() + datetime.timedelta(days=1)
    start_time = tomorrow.replace(hour=10, minute=0, second=0)
    end_time = start_time + datetime.timedelta(minutes=30) # 30 min meeting
    
    # 2. Format for Google (YYYYMMDDTHHMMSS)
    fmt = "%Y%m%dT%H%M%S"
    dates = f"{start_time.strftime(fmt)}/{end_time.strftime(fmt)}"
    
    # 3. Create Event Details
    title = f"🚨 SAVE: High Risk Customer ({data['tenure']}m Tenure)"
    details = f"URGENT RETENTION CALL\n\nRisk Factors:\n- Monthly Bill: ${data['MonthlyCharges']}\n- Rage Clicks: {data['Support_Ticket_Clicks']}\n\nGoal: Offer 20% discount."
    
    # 4. Build URL
    params = {
        "action": "TEMPLATE",
        "text": title,
        "details": details,
        "dates": dates,
        "ctz": "Asia/Kolkata" # Set to your timezone
    }
    return f"https://calendar.google.com/calendar/render?{urllib.parse.urlencode(params)}"

# ==========================================
# 4. SLACK FUNCTION
# ==========================================
def send_slack_alert(data, risk_score):
    # PASTE YOUR WORKING WEBHOOK URL HERE
    webhook_url = "https://hooks.slack.com/services/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"
    
    # GENERATE LINKS
    email_link = generate_email_link(data)
    calendar_link = generate_calendar_link(data)

    message = {
        "text": "🚨 *System Alert*",
        "blocks": [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"🚨 *High Risk Customer Detected*\nRisk: *{risk_score:.1%}*\nMonthly Charges: ${data['MonthlyCharges']}"
                }
            },
            {
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "📧 Draft Email"},
                        "style": "primary",
                        "url": email_link
                    },
                    {
                        "type": "button",
                        "text": {"type": "plain_text", "text": "📅 Book Emergency Meeting"},
                        "style": "danger",
                        "url": calendar_link # <--- NEW FEATURE
                    }
                ]
            }
        ]
    }
    
    try:
        requests.post(webhook_url, json=message)
        st.success("✅ Slack Alert Sent (Check for the Calendar Button!)")
    except Exception as e:
        st.error(f"❌ Connection Error: {e}")

# ==========================================
# 5. FRONTEND UI
# ==========================================
st.title("🛡️ ChurnGuardian: AI Retention System")

if model is None:
    st.stop()

# INPUTS
col1, col2 = st.columns(2)
with col1:
    tenure = st.slider("Tenure", 0, 72, 1)
    monthly = st.number_input("Monthly Charges", value=100.0)
    total = st.number_input("Total Charges", value=100.0)
with col2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
    internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
    emails = st.slider("Unanswered Emails", 0, 10, 5)
    clicks = st.slider("Rage Clicks", 0, 20, 15)

# PREDICTION
if st.button("🚀 Analyze Risk"):
    input_dict = {
        "tenure": tenure, "MonthlyCharges": monthly, "TotalCharges": total,
        "Contract": contract, "PaymentMethod": payment, "InternetService": internet,
        "Unanswered_Emails": emails, "Support_Ticket_Clicks": clicks
    }
    
    # Process & Predict
    input_df = pd.DataFrame([input_dict])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    prob = model.predict_proba(input_df)[0][1]
    
    st.metric("Churn Risk Score", f"{prob:.1%}")
    
    # Force Alert for Demo
    st.markdown("---")
    st.warning("⚠️ High Risk Detected. Triggering Auto-Agents...")
    send_slack_alert(input_dict, prob)