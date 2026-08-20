import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="RecoverAI",
    page_icon="💰",
    layout="wide"
)

# -----------------------------
# Generate realistic synthetic data
# -----------------------------
np.random.seed(42)

n = 1000

amount = np.random.randint(200, 10000, n)
previous_success = np.random.binomial(1, 0.65, n)
failed_attempts = np.random.randint(1, 6, n)
customer_engaged = np.random.binomial(1, 0.60, n)

# Synthetic recovery pattern
score = (
    1.2 * previous_success
    + 1.0 * customer_engaged
    - 0.45 * failed_attempts
    - 0.00005 * amount
    + np.random.normal(0, 0.5, n)
)

recovered = (score > 0.25).astype(int)

data = pd.DataFrame({
    "amount": amount,
    "previous_success": previous_success,
    "failed_attempts": failed_attempts,
    "customer_engaged": customer_engaged,
    "recovered": recovered
})

features = [
    "amount",
    "previous_success",
    "failed_attempts",
    "customer_engaged"
]

X = data[features]
y = data["recovered"]

# -----------------------------
# Train ML model
# -----------------------------
model = LogisticRegression()
model.fit(X, y)

# -----------------------------
# Header
# -----------------------------
st.title("💰 RecoverAI")
st.subheader("AI-Powered Revenue Recovery Agent")

st.write(
    "RecoverAI helps merchants identify failed payments with high "
    "recovery potential, estimate recoverable revenue, and recommend "
    "the next best recovery action."
)

st.caption(
    "Prototype using synthetic transaction data — no real customer or "
    "payment information is used."
)

st.divider()

# -----------------------------
# Dashboard calculations
# -----------------------------
failed_transactions = len(data)

potential_revenue = int(
    data.loc[data["recovered"] == 0, "amount"].sum()
)

overall_recovery_rate = data["recovered"].mean() * 100

predicted_probabilities = model.predict_proba(data[features])[:, 1]

expected_revenue = int(
    np.sum(data["amount"] * predicted_probabilities)
)

# -----------------------------
# Dashboard
# -----------------------------
st.header("📊 Revenue Recovery Dashboard")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Transactions Analyzed",
    f"{failed_transactions:,}"
)

c2.metric(
    "Revenue at Risk",
    f"₹{potential_revenue:,.0f}"
)

c3.metric(
    "Recovery Rate",
    f"{overall_recovery_rate:.1f}%"
)

c4.metric(
    "Expected Recoverable Revenue",
    f"₹{expected_revenue:,.0f}"
)

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs([
    "🤖 AI Recovery Agent",
    "📈 Analytics",
    "💡 How It Works"
])

# =========================================================
# TAB 1 — AI RECOVERY
# =========================================================

with tab1:

    st.header("🤖 Analyze a Failed Payment")

    left, right = st.columns(2)

    with left:

        transaction_amount = st.number_input(
            "Transaction Amount (₹)",
            min_value=100,
            max_value=100000,
            value=2499,
            step=100
        )

        previous_payment = st.selectbox(
            "Previous successful payments?",
            ["Yes", "No"]
        )

    with right:

        failed_count = st.slider(
            "Number of failed attempts",
            min_value=1,
            max_value=10,
            value=1
        )

        engagement = st.selectbox(
            "Customer recently engaged?",
            ["Yes", "No"]
        )

    if st.button(
        "🚀 Analyze Recovery Opportunity",
        use_container_width=True
    ):

        previous_value = 1 if previous_payment == "Yes" else 0
        engagement_value = 1 if engagement == "Yes" else 0

        customer_input = pd.DataFrame([{
            "amount": transaction_amount,
            "previous_success": previous_value,
            "failed_attempts": failed_count,
            "customer_engaged": engagement_value
        }])

        probability = model.predict_proba(customer_input)[0][1]

        probability_percent = round(probability * 100)

        expected_revenue_customer = (
            transaction_amount * probability
        )

        # -----------------------------
        # Recovery strategy
        # -----------------------------

        if probability >= 0.75:

            priority = "HIGH"

            action = (
                "Send a personalized payment recovery message "
                "and provide an immediate retry option."
            )

            message = (
                f"Hi! We noticed that your payment of "
                f"₹{transaction_amount:,.0f} couldn't be completed. "
                "You can securely retry your payment now. "
                "We're happy to help if you face any issues."
            )

        elif probability >= 0.45:

            priority = "MEDIUM"

            action = (
                "Send a payment reminder and schedule one "
                "automated retry."
            )

            message = (
                f"Your payment of ₹{transaction_amount:,.0f} "
                "is still pending. Please retry your payment "
                "when convenient."
            )

        else:

            priority = "LOW"

            action = (
                "Avoid repeated payment attempts and route "
                "the customer to support."
            )

            message = (
                "We couldn't complete your recent payment. "
                "Our support team can help you complete the transaction."
            )

        # -----------------------------
        # Results
        # -----------------------------

        st.divider()

        r1, r2, r3 = st.columns(3)

        r1.metric(
            "Recovery Probability",
            f"{probability_percent}%"
        )

        r2.metric(
            "Expected Recoverable Revenue",
            f"₹{expected_revenue_customer:,.0f}"
        )

        r3.metric(
            "Priority",
            priority
        )

        st.subheader("🎯 Recommended Action")

        st.success(action)

        st.subheader("✉️ Personalized Recovery Message")

        st.text_area(
            "AI-generated message",
            message,
            height=130
        )

        st.subheader("🧠 Why did the AI make this decision?")

        reasons = []

        if previous_value:
            reasons.append(
                "Previous successful payments increase recovery confidence."
            )
        else:
            reasons.append(
                "No previous successful payment history reduces confidence."
            )

        if engagement_value:
            reasons.append(
                "Recent customer engagement is a positive recovery signal."
            )
        else:
            reasons.append(
                "Low recent engagement reduces recovery confidence."
            )

        if failed_count >= 3:
            reasons.append(
                "Multiple failed attempts reduce recovery confidence."
            )
        else:
            reasons.append(
                "A small number of failed attempts leaves room for recovery."
            )

        if transaction_amount > 5000:
            reasons.append(
                "Higher transaction value increases the amount at risk."
            )

        for reason in reasons:
            st.write("• " + reason)

# =========================================================
# TAB 2 — ANALYTICS
# =========================================================

with tab2:

    st.header("📈 Recovery Analytics")

    chart_data = pd.DataFrame({
        "Category": [
            "Recovered",
            "High Opportunity",
            "Medium Opportunity",
            "Low Opportunity"
        ],
        "Transactions": [
            int(data["recovered"].sum()),
            int(np.sum(predicted_probabilities >= 0.75)),
            int(np.sum(
                (predicted_probabilities >= 0.45)
                & (predicted_probabilities < 0.75)
            )),
            int(np.sum(predicted_probabilities < 0.45))
        ]
    })

    st.bar_chart(
        chart_data.set_index("Category")
    )

    st.subheader("🔎 Sample Recovery Opportunities")

    sample = data.head(10).copy()

    sample["Recovery Probability"] = (
        model.predict_proba(sample[features])[:, 1] * 100
    ).round(1)

    sample["Expected Revenue"] = (
        sample["amount"]
        * sample["Recovery Probability"]
        / 100
    ).round(0)

    display_data = sample[
        [
            "amount",
            "failed_attempts",
            "Recovery Probability",
            "Expected Revenue"
        ]
    ].copy()

    display_data.columns = [
        "Amount (₹)",
        "Failed Attempts",
        "Recovery Probability (%)",
        "Expected Revenue (₹)"
    ]

    st.dataframe(
        display_data,
        use_container_width=True,
        hide_index=True
    )

# =========================================================
# TAB 3 — HOW IT WORKS
# =========================================================

with tab3:

    st.header("💡 How RecoverAI Works")

    st.write("### 1️⃣ Detect")

    st.write(
        "The system receives information about a failed payment."
    )

    st.write("### 2️⃣ Predict")

    st.write(
        "A machine-learning model estimates the probability "
        "of successfully recovering the payment."
    )

    st.write("### 3️⃣ Estimate")

    st.write(
        "RecoverAI combines transaction value and recovery probability "
        "to estimate expected recoverable revenue."
    )

    st.write("### 4️⃣ Decide")

    st.write(
        "The agent selects an appropriate recovery strategy "
        "based on the customer's situation."
    )

    st.write("### 5️⃣ Engage")

    st.write(
        "A personalized recovery message is generated for the customer."
    )

    st.write("### 6️⃣ Recover")

    st.write(
        "The merchant can use the recommended action to attempt "
        "to recover otherwise-lost revenue."
    )

    st.divider()

    st.info(
        "RecoverAI is a prototype built with synthetic transaction data "
        "to demonstrate AI-assisted revenue recovery."
    )

st.divider()

st.caption(
    "RecoverAI | AI Revenue Recovery Prototype"
)