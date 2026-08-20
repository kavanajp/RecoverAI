# RecoverAI – AI-Powered Revenue Recovery Agent

RecoverAI is an AI-assisted revenue recovery prototype designed to help merchants identify failed payment opportunities, estimate potential recoverable revenue, and recommend suitable recovery actions.

## Problem

Failed payments can result in lost revenue for businesses. Not every failed transaction should be treated the same way. Some customers may have a high probability of completing the payment, while repeated retries for other customers may not be effective.

## Solution

RecoverAI uses machine learning to:

- Analyze failed payment information
- Predict recovery probability
- Estimate expected recoverable revenue
- Prioritize recovery opportunities
- Recommend a suitable recovery action
- Generate a personalized customer recovery message

## How It Works

1. Merchant enters transaction and customer information.
2. The ML model analyzes the input.
3. RecoverAI predicts the probability of payment recovery.
4. Expected recoverable revenue is calculated.
5. The system recommends the next recovery action.
6. A personalized recovery message is generated.

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Logistic Regression

## Features

- AI recovery probability prediction
- Expected recoverable revenue
- Recovery priority
- Personalized recovery message
- Recovery analytics dashboard
- What-if analysis
- AI decision explanation

## Running the Project

Install the required dependencies:

```bash
pip install -r requirements.txt