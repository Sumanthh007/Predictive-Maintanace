import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===============================
# LOAD DATA (VERY IMPORTANT)
# ===============================
@st.cache_data
def load_data():
    train = pd.read_csv('train_FD004.txt', sep=r'\s+', header=None)
    test = pd.read_csv('test_FD004.txt', sep=r'\s+', header=None)
    rul_test = pd.read_csv('RUL_FD004.txt', header=None)

    columns = (
        ['engine_id', 'cycle'] +
        [f'op_setting_{i}' for i in range(1,4)] +
        [f'sensor_{i}' for i in range(1,22)]
    )

    train.columns = columns
    test.columns = columns
    rul_test.columns = ['RUL']

    return train, test, rul_test

train, test, rul_test = load_data()

# ===============================
# FEATURE COLUMNS
# ===============================
feature_cols = [col for col in train.columns if col not in ['engine_id','cycle']]

WINDOW_SIZE = 30

# ===============================
# DUMMY MODEL (IMPORTANT FIX)
# ===============================
# 👉 Replace this later with your real model if needed
from tensorflow.keras.models import load_model
import tensorflow as tf

# Custom functions (VERY IMPORTANT)
def rul_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    error = tf.abs(y_true - y_pred)
    return tf.reduce_mean(tf.cast(error <= 20, tf.float32))

def asymmetric_weighted_mae(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    error = y_pred - y_true
    abs_error = tf.abs(error)

    weight = 1.0 / (y_true + 1.0)
    over_penalty = tf.where(error > 0, 2.0, 1.0)

    return tf.reduce_mean(weight * over_penalty * abs_error)

# Load model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras import regularizers

model = Sequential([
    Input(shape=(30, 24)),

    LSTM(64, return_sequences=True,
         kernel_regularizer=regularizers.l2(1e-4),
         recurrent_regularizer=regularizers.l2(1e-4)),
    Dropout(0.4),

    LSTM(32, return_sequences=False,
         kernel_regularizer=regularizers.l2(1e-4),
         recurrent_regularizer=regularizers.l2(1e-4)),
    Dropout(0.4),

    Dense(32, activation='relu',
          kernel_regularizer=regularizers.l2(1e-4)),

    BatchNormalization(),

    Dense(1)
])
model.load_weights("model.weights.h5")
# ===============================
# SIMPLE SCALER (SAFE FIX)
# ===============================
import joblib
scaler = joblib.load("scaler.pkl")

# ===============================
# WEIBULL FUNCTIONS (SAFE)
# ===============================
def pdf(t, beta, eta):
    return (beta/eta) * (t/eta)**(beta-1) * np.exp(-(t/eta)**beta)

def cdf(t, beta, eta):
    return 1 - np.exp(-(t/eta)**beta)

def hazard(t, beta, eta):
    return pdf(t, beta, eta) / (1 - cdf(t, beta, eta) + 1e-6)

# Dummy Weibull params
beta = 2.0
eta = 200
failure_times = np.linspace(0, 300, 100)

# ===============================
# TITLE
# ===============================
st.title("Predictive Maintenance Dashboard")

# ===============================
# SELECT ENGINE
# ===============================
engine_id = st.number_input(
    "Select Engine ID",
    min_value=1,
    max_value=int(test['engine_id'].max()),
    value=1,
    step=1
)

# ===============================
# ENGINE DATA
# ===============================
engine_data = test[test['engine_id'] == engine_id].sort_values('cycle')

if engine_data.empty:
    st.error("Invalid Engine ID")
    st.stop()

current_cycle = engine_data['cycle'].max()

# ===============================
# PREPARE INPUT
# ===============================
features = engine_data[feature_cols].values

if len(features) >= WINDOW_SIZE:
    seq = features[-WINDOW_SIZE:]
else:
    padding = np.zeros((WINDOW_SIZE - len(features), features.shape[1]))
    seq = np.vstack((padding, features))

seq = seq.reshape(1, WINDOW_SIZE, len(feature_cols))

# Scale
seq_2d = seq.reshape(-1, seq.shape[-1])
seq_scaled = scaler.transform(seq_2d).reshape(seq.shape)

# ===============================
# PREDICTION
# ===============================
predicted_rul = model.predict(seq_scaled).flatten()[0]
predicted_failure_cycle = current_cycle + predicted_rul

true_rul = rul_test.iloc[engine_id - 1]['RUL']
true_failure_cycle = current_cycle + true_rul

# ===============================
# TEXT OUTPUT
# ===============================
st.subheader("ENGINE RELIABILITY ANALYSIS")

st.write(f"Engine ID: {engine_id}")
st.write(f"Current observed cycle: {current_cycle}")

st.subheader("RUL Prediction")
st.write(f"True RUL: {true_rul}")
st.write(f"Predicted RUL: {predicted_rul:.2f}")
st.write(f"Absolute Error: {abs(true_rul - predicted_rul):.2f}")

st.subheader("Failure Cycle Estimation")
st.write(f"True failure cycle: {true_failure_cycle}")
st.write(f"Predicted failure cycle: {predicted_failure_cycle:.2f}")

# ===============================
# DECISION
# ===============================
st.subheader("Maintenance Decision")

if predicted_rul < 25:
    st.error("⚠ Maintenance Recommended")
else:
    st.success("✓ Engine Safe For Next Operation Window")

# ===============================
# WEIBULL RELIABILITY PLOT (MATCH NOTEBOOK)
# ===============================

t = np.linspace(0, max(failure_times)+50, 500)

pdf_vals = pdf(t, beta, eta)
cdf_vals = cdf(t, beta, eta)
hazard_vals = hazard(t, beta, eta)

fig, ax1 = plt.subplots(figsize=(12,7))

# ✅ SAME COLORS AS NOTEBOOK
line1, = ax1.plot(t, pdf_vals, color='blue', linewidth=2, label="Weibull PDF")
line2, = ax1.plot(t, cdf_vals, color='green', linewidth=2, label="Weibull CDF")

line3 = ax1.axvline(current_cycle, color='black', linestyle='--', linewidth=2, label="Current Cycle")
line4 = ax1.axvline(predicted_failure_cycle, color='red', linestyle='--', linewidth=2, label="Predicted Failure")
line5 = ax1.axvline(true_failure_cycle, color='orange', linestyle='--', linewidth=2, label="True Failure")
line6 = ax1.axvline(eta, color='purple', linestyle=':', linewidth=2, label="Characteristic Life (η)")

ax1.set_xlabel("Cycle")
ax1.set_ylabel("PDF / CDF")
ax1.grid(True, linestyle='--', alpha=0.6)

# ✅ Hazard curve SAME color
ax2 = ax1.twinx()
line7, = ax2.plot(t, hazard_vals, color='darkcyan', linewidth=2, label="Hazard Rate")
ax2.set_ylabel("Hazard Rate")

# ✅ SAME legend style
lines = [line1, line2, line3, line4, line5, line6, line7]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="upper left")

# ✅ SAME annotation
ax1.text(
    current_cycle + 10,
    0.6,
    f"Predicted RUL ≈ {predicted_rul:.1f}\nTrue RUL = {true_rul}",
    bbox=dict(facecolor='white', alpha=0.9)
)

plt.title(f"Unified Reliability Analysis - Engine {engine_id}")

# ✅ IMPORTANT for Streamlit
st.pyplot(fig)