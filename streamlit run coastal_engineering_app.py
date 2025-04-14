import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
from fpdf import FPDF
import PyCO2SYS as pyco2


# --- Background Styling ---
def set_ocean_background():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1507525428034-b723cf961d3e");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        .main {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 1rem;
            border-radius: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Core Functions ---
def compute_wave_energy(Hs, Tp):
    rho = 1025  # density of seawater (kg/m³)
    g = 9.81    # gravity (m/s²)
    E = (1/8) * rho * g * Hs**2
    Cg = g * Tp / (4 * np.pi)
    return E * Cg

def estimate_sediment_transport(Hs, angle, Tp):
    K = 0.77
    return K * Hs**2 * np.sin(2 * np.radians(angle)) * Tp

def predict_shoreline_change(Q_sed):
    dx = 100  # grid spacing in meters
    dQdx = np.gradient(Q_sed, dx)
    return -dQdx

def carbonate_impact(TA, DIC, S=35, T=25, P=0):
    results = pyco2.sys(
        par1=TA, par2=DIC,
        par1_type=1, par2_type=2,
        salinity=S, temperature=T, pressure=P,
        opt_pH_scale=1
    )
    omega = results["output"]["OmegaAR"]
    return omega[0] if isinstance(omega, (list, np.ndarray)) else omega

# --- PDF Report Generator ---
def create_pdf(Hs, Tp, angle, TA, DIC, omega_value, fig):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Coastal & Ocean Engineering Report", ln=True, align="C")

    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Wave Height (Hs): {Hs} m", ln=True)
    pdf.cell(200, 10, txt=f"Wave Period (Tp): {Tp} s", ln=True)
    pdf.cell(200, 10, txt=f"Wave Angle: {angle}°", ln=True)
    pdf.cell(200, 10, txt=f"Total Alkalinity (TA): {TA} µmol/kg", ln=True)
    pdf.cell(200, 10, txt=f"Dissolved Inorganic Carbon (DIC): {DIC} µmol/kg", ln=True)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 10, txt=f"Aragonite Saturation State (Ωₐ): {omega_value:.2f}", ln=True)

    # Save plot to buffer and insert as image
    buf = BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    pdf.image(buf, x=10, y=pdf.get_y() + 10, w=180)
    buf.close()

    return pdf.output(dest="S").encode("latin-1")

# --- App Start ---
set_ocean_background()

st.title("🌊 Coastal & Ocean Engineering App")
st.markdown("""
This app models **wave energy**, **sediment transport**, **shoreline change**, and estimates the **carbonate chemistry** impact using your inputs.
""")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("🌊 Wave Parameters")
    Hs = st.slider("Significant Wave Height (m)", 0.5, 5.0, 2.0, step=0.1)
    Tp = st.slider("Peak Wave Period (s)", 4, 20, 8, step=1)
    angle = st.slider("Wave Angle (degrees)", 0, 90, 20, step=1)

    st.header("🧪 Carbonate Chemistry")
    TA = st.number_input("Total Alkalinity (µmol/kg)", min_value=2000, max_value=2500, value=2300)
    DIC = st.number_input("Dissolved Inorganic Carbon (µmol/kg)", min_value=1800, max_value=2300, value=2100)

# --- Calculations ---
Hs_array = np.linspace(max(0.1, Hs - 0.5), Hs + 0.5, 20)
Tp_array = np.full_like(Hs_array, Tp)
angle_array = np.linspace(max(0, angle - 10), min(angle + 10, 90), 20)

wave_energy = compute_wave_energy(Hs_array, Tp_array)
Q_sed = estimate_sediment_transport(Hs_array, angle_array, Tp_array)
shoreline_change = predict_shoreline_change(Q_sed)

try:
    omega_arag = carbonate_impact(TA, DIC)
except Exception as e:
    omega_arag = None
    st.error(f"Error computing carbonate system: {e}")

# --- Plot Results ---
st.subheader("📉 Predicted Shoreline Change")
fig, ax = plt.subplots()
ax.plot(shoreline_change, color='teal', label="Shoreline Change Rate")
ax.set_xlabel("Grid Index")
ax.set_ylabel("Shoreline Change Rate (m/s)")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# --- Aragonite Result ---
st.subheader("🡢 Aragonite Saturation State (Ωₐ)")
if omega_arag is not None:
    st.metric("Ωₐ", f"{omega_arag:.2f}")
else:
    st.warning("Could not compute Ωₐ with the current inputs.")

# --- CSV Export ---
st.subheader("📅 Export Results")

data = pd.DataFrame({
    "Hs (m)": Hs_array,
    "Wave Energy (J/m²)": wave_energy,
    "Sediment Transport": Q_sed,
    "Shoreline Change Rate (m/s)": shoreline_change
})

csv = data.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download CSV", data=csv, file_name="coastal_results.csv", mime="text/csv")

# --- PDF Export ---
if omega_arag is not None:
    pdf_bytes = create_pdf(Hs, Tp, angle, TA, DIC, omega_arag, fig)
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name="coastal_report.pdf",
        mime="application/pdf"
    )

# --- Footer Note ---
st.markdown(
    "This model is simplified and intended for educational or demonstration purposes only. "
    "Run it live at [Streamlit Cloud](https://streamlit.io/cloud)."
)