import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import poisson, norm
import matplotlib.animation as animation
import tempfile

st.set_page_config(page_title="Poissonverteilung – Analyse", layout="wide")
st.title("Poissonverteilung – Interaktive Analyse & Visualisierung")

tabs = st.tabs([
    "📊 Interaktiv: PMF, CDF, CCDF",
    "🔁 Normalapproximation",
    "📈 Animation (Plotly)",
    "🎞️ Animation (Matplotlib-GIF)"
])

# === TAB 1 ===
with tabs[0]:
    st.subheader("📊 Interaktive Darstellung der Poissonverteilung")

    lambda_val = st.sidebar.slider("λ (Erwartungswert)", 0.1, 100.0, 4.0, 0.1, key="lambda_tab1")
    x_min = st.sidebar.number_input("x-Minimum", 0, 1000, 0, step=1, key="x_min_tab1")
    x_max = st.sidebar.number_input("x-Maximum", 1, 1000, 15, step=1, key="x_max_tab1")
    k = st.sidebar.number_input("k-Wert", 0, 1000, 2, step=1, key="k_tab1")

    if x_min >= x_max:
        st.error("x-Minimum muss kleiner als x-Maximum sein.")
    else:
        x = np.arange(x_min, x_max + 1)
        pmf_y = poisson.pmf(x, mu=lambda_val)
        cdf_y = poisson.cdf(x, mu=lambda_val)
        ccdf_y = 1 - cdf_y + pmf_y

        bar_colors = ['royalblue' if val <= k else 'skyblue' for val in x]

        # PMF
        fig1, ax1 = plt.subplots()
        ax1.bar(x, pmf_y, color=bar_colors, edgecolor='black')
        if x_min <= k <= x_max:
            ax1.bar(k, poisson.pmf(k, mu=lambda_val), color='orange', edgecolor='black', label=f"P(X = {k})")
            ax1.legend()
        ax1.set_title(f"Wahrscheinlichkeitsfunktion (λ = {lambda_val})")
        ax1.set_xlabel("x")
        ax1.set_ylabel("P(X = x)")
        ax1.grid(True, linestyle='--', alpha=0.5)
        st.pyplot(fig1)

        # CDF
        fig2, ax2 = plt.subplots()
        ax2.plot(x, cdf_y, marker='o', linestyle='-', color='green', label="P(X ≤ x)")
        ax2.axvline(k, color='orange', linestyle='--', label=f"x = {k}")
        ax2.axhline(poisson.cdf(k, mu=lambda_val), color='gray', linestyle=':', label=f"P(X ≤ {k}) = {poisson.cdf(k, mu=lambda_val):.4f}")
        ax2.set_title("Kumulative Verteilung")
        ax2.set_xlabel("x")
        ax2.set_ylabel("P(X ≤ x)")
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()
        st.pyplot(fig2)

        # CCDF
        fig3, ax3 = plt.subplots()
        ax3.plot(x, ccdf_y, marker='o', linestyle='-', color='purple', label="P(X ≥ x)")
        ax3.axvline(k, color='orange', linestyle='--', label=f"x = {k}")
        if x_min <= k <= x_max:
            ax3.axhline(ccdf_y[k - x_min], color='gray', linestyle=':', label=f"P(X ≥ {k}) = {ccdf_y[k - x_min]:.4f}")
        ax3.set_title("Komplementäre kumulative Verteilung")
        ax3.set_xlabel("x")
        ax3.set_ylabel("P(X ≥ x)")
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.legend()
        st.pyplot(fig3)

        # Wahrscheinlichkeiten für k
        st.subheader("Wahrscheinlichkeiten für k")
        prob_k = poisson.pmf(k, mu=lambda_val)
        cdf_k = poisson.cdf(k, mu=lambda_val)
        ccdf_k = 1 - cdf_k + prob_k
        st.write(f"**P(X = {k}) = {prob_k:.4f}**")
        st.write(f"**P(X ≤ {k}) = {cdf_k:.4f}**")
        st.write(f"**P(X ≥ {k}) = {ccdf_k:.4f}**")

        # Tabelle
        st.subheader("Tabelle: Wahrscheinlichkeiten")
        st.dataframe({
            "x": x,
            f"P(X = x) (λ = {lambda_val})": np.round(pmf_y, 6),
            f"P(X ≤ x)": np.round(cdf_y, 6),
            f"P(X ≥ x)": np.round(ccdf_y, 6),
        })

# === TAB 2 ===
with tabs[1]:
    st.subheader("🔁 Poisson-Verteilung mit Normalapproximation")

    lambda_val = st.sidebar.number_input("λ (Erwartungswert)", 0.1, 100.0, 4.0, step=0.1, key="lambda_tab2")
    x_min = st.sidebar.number_input("x-Minimum", 0, 1000, 0, step=1, key="x_min_tab2")
    x_max = st.sidebar.number_input("x-Maximum", 1, 1000, 15, step=1, key="x_max_tab2")
    k = st.sidebar.number_input("k-Wert", 0, 1000, 2, step=1, key="k_tab2")

    if x_min >= x_max:
        st.error("x-Minimum muss kleiner als x-Maximum sein.")
    else:
        x = np.arange(x_min, x_max + 1)
        y_poisson = poisson.pmf(x, mu=lambda_val)

        mu = lambda_val
        sigma = np.sqrt(lambda_val)
        x_dense = np.linspace(x_min - 0.5, x_max + 0.5, 1000)
        y_normal = norm.pdf(x_dense, loc=mu, scale=sigma)

        fig, ax = plt.subplots()
        ax.bar(x, y_poisson, color='skyblue', edgecolor='black', label="Poisson P(X=x)")
        ax.plot(x_dense, y_normal, color='red', lw=2, label="Normalapproximation")
        if x_min <= k <= x_max:
            ax.bar(k, poisson.pmf(k, mu=lambda_val), color='orange', edgecolor='black', label=f"P(X={k})")
        ax.set_title(f"Poisson vs. Normal (λ = {lambda_val})")
        ax.set_xlabel("x")
        ax.set_ylabel("Wahrscheinlichkeit")
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        st.pyplot(fig)

        st.subheader("Wahrscheinlichkeiten")
        st.dataframe({
            "x": x,
            f"P(X = x) (Poisson, λ={lambda_val})": np.round(y_poisson, 4)
        })

        st.subheader(f"P(X = {k})")
        prob_k = poisson.pmf(k, mu=lambda_val)
        st.write(f"**P(X = {k}) = {prob_k:.4f}**")

# === TAB 3 ===
with tabs[2]:
    st.subheader("📈 Animation der Verteilung (Plotly)")

    x_min = st.sidebar.number_input("x-Minimum", 0, 1000, 0, step=1, key="x_min_tab3")
    x_max = st.sidebar.number_input("x-Maximum", 1, 1000, 80, step=1, key="x_max_tab3")

    if x_min >= x_max:
        st.error("x-Minimum muss kleiner als x-Maximum sein.")
    else:
        x_vals = np.arange(x_min, x_max + 1)
        lambda_values = np.linspace(1.1, 60.0, 100)

        y_init = poisson.pmf(x_vals, mu=lambda_values[0])
        fig = go.Figure(
            data=[go.Bar(x=x_vals, y=y_init, marker_color='skyblue')],
            layout=go.Layout(
                title=f"Poisson-Verteilung (λ = {lambda_values[0]:.1f})",
                xaxis=dict(title="x"),
                yaxis=dict(title="P(X = x)", range=[0, 0.3]),
                updatemenus=[dict(
                    type="buttons",
                    showactive=False,
                    buttons=[dict(label="▶️ Start", method="animate", args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}])]
                )]
            ),
            frames=[
                go.Frame(
                    data=[go.Bar(x=x_vals, y=poisson.pmf(x_vals, mu=lmb), marker_color='skyblue')],
                    layout=go.Layout(title_text=f"Poisson-Verteilung (λ = {lmb:.1f})")
                )
                for lmb in lambda_values
            ]
        )

        st.plotly_chart(fig, use_container_width=True)

# === TAB 4 ===
with tabs[3]:
    st.subheader("🎞️ Matplotlib-Animation als GIF")

    x_min = st.sidebar.number_input("x-Minimum", 0, 1000, 0, step=1, key="x_min_tab4")
    x_max = st.sidebar.number_input("x-Maximum", 1, 1000, 80, step=1, key="x_max_tab4")

    if x_min >= x_max:
        st.error("x-Minimum muss kleiner als x-Maximum sein.")
    else:
        fig, ax = plt.subplots()

        def animate(frame):
            ax.clear()
            lambda_val = frame
            x = np.arange(x_min, x_max + 1)
            y = poisson.pmf(x, mu=lambda_val)
            ax.bar(x, y, color='skyblue', edgecolor='black')
            ax.set_title(f"Poisson-Verteilung (λ = {lambda_val:.1f})")
            ax.set_xlabel("x")
            ax.set_ylabel("P(X = x)")
            ax.set_ylim(0, 0.25)
            ax.grid(True, linestyle='--', alpha=0.5)

        lambda_frames = np.linspace(1.1, 60, 100)
        ani = animation.FuncAnimation(fig, animate, frames=lambda_frames, interval=150)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as tmpfile:
            ani.save(tmpfile.name, writer="pillow")
            gif_path = tmpfile.name

        st.image(gif_path, caption="Animation der Poisson-Verteilung für λ von 1.1 bis 60", use_container_width=True)
