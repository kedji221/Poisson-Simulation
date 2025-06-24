import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

st.set_page_config(page_title="Poissonverteilung", layout="centered")

# Titel
st.title("Poissonverteilung interaktiv")

# Sidebar mit Eingabeparametern
st.sidebar.header("Parameter")
lambda_val = st.sidebar.slider("λ (Erwartungswert)", min_value=0.1, max_value=100.0, value=4.0, step=0.1)
x_min = st.sidebar.number_input("x-Minimum", min_value=0, value=0, step=1)
x_max = st.sidebar.number_input("x-Maximum", min_value=1, value=15, step=1)
k = st.sidebar.number_input("k-Wert für P(X = k), P(X ≤ k), P(X ≥ k)", min_value=0, value=2, step=1)

if x_min >= x_max:
    st.error("x-Minimum muss kleiner als x-Maximum sein.")
else:
    # Werte berechnen
    x = np.arange(x_min, x_max + 1)
    pmf_y = poisson.pmf(x, mu=lambda_val)
    cdf_y = poisson.cdf(x, mu=lambda_val)
    ccdf_y = 1 - cdf_y + pmf_y  # P(X ≥ x) korrekt berechnet

    # Farben für Balkendiagramm
    bar_colors = ['royalblue' if val <= k else 'skyblue' for val in x]

    # Plot 1: PMF
    fig1, ax1 = plt.subplots()
    ax1.bar(x, pmf_y, color=bar_colors, edgecolor='black')
    if x_min <= k <= x_max:
        ax1.bar(k, poisson.pmf(k, mu=lambda_val), color='orange', edgecolor='black', label=f"P(X = {k})")
        ax1.legend()
    ax1.set_title(f"Wahrscheinlichkeitsfunktion (genau) (λ = {lambda_val})")
    ax1.set_xlabel("x")
    ax1.set_ylabel("P(X = x)")
    ax1.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig1)

    # Plot 2: CDF
    fig2, ax2 = plt.subplots()
    ax2.plot(x, cdf_y, marker='o', linestyle='-', color='green', label="P(X ≤ x)")
    ax2.axvline(k, color='orange', linestyle='--', label=f"x = {k}")
    ax2.axhline(poisson.cdf(k, mu=lambda_val), color='gray', linestyle=':', label=f"P(X ≤ {k}) = {poisson.cdf(k, mu=lambda_val):.4f}")
    ax2.set_title("Kumulative Verteilung (P(X ≤ x))")
    ax2.set_xlabel("x")
    ax2.set_ylabel("P(X ≤ x)")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    st.pyplot(fig2)

    # Plot 3: P(X ≥ x)
    fig3, ax3 = plt.subplots()
    ax3.plot(x, ccdf_y, marker='o', linestyle='-', color='purple', label="P(X ≥ x)")
    ax3.axvline(k, color='orange', linestyle='--', label=f"x = {k}")
    ax3.axhline(ccdf_y[k - x_min] if x_min <= k <= x_max else 0, color='gray', linestyle=':', label=f"P(X ≥ {k}) = {ccdf_y[k - x_min]:.4f}")
    ax3.set_title("Komplementäre kumulative Verteilung (P(X ≥ x))")
    ax3.set_xlabel("x")
    ax3.set_ylabel("P(X ≥ x)")
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend()
    st.pyplot(fig3)

    # Wahrscheinlichkeiten für k anzeigen
    st.subheader("Wahrscheinlichkeiten für k")
    prob_k = poisson.pmf(k, mu=lambda_val)
    cdf_k = poisson.cdf(k, mu=lambda_val)
    ccdf_k = 1 - cdf_k + prob_k
    st.write(f"**P(X = {k}) = {prob_k:.4f}**")
    st.write(f"**P(X ≤ {k}) = {cdf_k:.4f}**")
    st.write(f"**P(X ≥ {k}) = {ccdf_k:.4f}**")

    # Erweiterte Tabelle: 3-in-1
    st.subheader("Tabelle: Wahrscheinlichkeiten")
    table_combined = {
        "x": x,
        f"P(X = x) bei λ = {lambda_val}": np.round(pmf_y, 6),
        f"P(X ≤ x) bei λ = {lambda_val}": np.round(cdf_y, 6),
        f"P(X ≥ x) bei λ = {lambda_val}": np.round(ccdf_y, 6)
    }
    st.dataframe(table_combined)
