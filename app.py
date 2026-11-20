import streamlit as st
import pandas as pd
import numpy as np
import io
from PIL import Image

# -----------------------------------------------------------------------------
# 1. Dataset Initialization
# -----------------------------------------------------------------------------
raw_data = """Product Name,Size (DN),Differential Pressure (bar g),Orifice Size (mm)
SOFT31,15,4.5,4
SOFT31,15,10,2.6
SOFT31,15,13,2
SOFT31,20,4.5,4
SOFT31,20,10,2.6
SOFT31,20,13,2
SOFT31,25,4.5,5.3
SOFT31,25,10,3.1
SOFT31,40,4.5,18
SOFT31,40,10,15.5
SOFT31,40,14,14
SOFT31,40,21,14
SOFT31,50,4.5,29
SOFT31,50,10,21
SOFT31,50,14,17
SOFT31,50,21,17
SOFT53,15,4.5,4
SOFT53,15,10,3.2
SOFT53,15,14,2.7
SOFT53,15,21,2
SOFT53,15,32,1.6
SOFT53,20,4.5,4
SOFT53,20,10,3.2
SOFT53,20,14,2.7
SOFT53,20,21,2
SOFT53,20,32,1.6
SOFT53,25,4.5,7
SOFT53,25,10,5.7
SOFT53,25,14,4.7
SOFT53,25,21,4
SOFT53,25,32,3
SOFT53,40,4.5,18
SOFT53,40,10,15.5
SOFT53,40,14,14
SOFT53,40,21,14
SOFT53,40,32,14
SOFT53,50,4.5,29
SOFT53,50,10,21
SOFT53,50,14,17
SOFT53,50,21,17
SOFT53,50,32,3
SOFT 53 X 100,100,4.5,29
SOFT 53 X 100,100,10,21
SOFT 53 X 100,100,14,17
SOFT 53 X 100,100,21,17
SOFT 53 X 100,100,32,17
CMTOFT,15,4.5,4
CMTOFT,15,10,2.6
CMTOFT,15,15,2
CMTOFT,20,4.5,4
CMTOFT,20,10,2.6
CMTOFT,20,15,2
SOFT54,25,42,1.6
SOFT31-O,15,13,2
FMTD60-O,15,-,4.25
FMTD65,15,-,3.2
FMTD65,20,-,3.2
FMTD722-M,15,-,3
FMTD722-M,20,-,3.2
FMTD76,15,-,3.5
FMTD76,20,-,3.5
FMTD64,15,-,4.4
FMTD64,20,-,5
FMTD25,6,-,2.8
FMTD64-U,-,-,4.4
CMTD62M,15,-,3.5
CMTD62M,20,-,3.5
CMTD42M,15,-,5
CMTD42M,20,-,5
GTRAP,6,-,1.2
GTRAP,10,-,1.2
FMMST63,15,-,2
FMAV53,15,-,4.5
FMAV53,20,-,4.5
FMAV61,15,-,4.5
FMAV61,20,-,4.5
FMTLT63,15,-,4.5
FMTLT63,20,-,4.5
FMTLT61,15,-,4.5
FMTLT61,20,-,4.5
FMLDT31-O,15,13,2
FMLDT31,15,13,2
FMLDT31,20,13,2
FMLDT31,25,13,3.3
FMLDT53,15,4.5,4
FMLDT53,15,10,3.2
FMLDT53,15,14,2.7
FMLDT53,15,21,2
FMLDT53,20,4.5,4
FMLDT53,20,10,3.2
FMLDT53,20,14,2.7
FMLDT53,20,21,2
FMLDT53,25,4.5,7
FMLDT53,25,10,7
FMLDT53,25,14,5.7
FMLDT53,25,21,4.7
Float trap,15,4.5,4
Float trap,15,10,3.2
Float trap,15,14,2.7
Float trap,15,21,2
Float trap,15,32,1.6
Float trap,20,4.5,4
Float trap,20,10,3.2
Float trap,20,14,2.7
Float trap,20,21,2
Float trap,20,32,1.6"""

df = pd.read_csv(io.StringIO(raw_data))

# -----------------------------------------------------------------------------
# 2. UI Setup & Theory
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Steam Trap Loss Calculator", layout="wide")
st.title("Steam Trap Loss Calculator & Modeler")

st.header("1. Theory & Nomenclature")
col1, col2 = st.columns(2)
with col1:
    st.subheader("Napier's Equation")
    try:
        st.image("image_3ea26b.jpg", caption="Live saturated steam flow through an orifice (choked flow)")
    except Exception as e:
        st.warning("Please upload 'image_3ea26b.jpg' to the working directory.")
        
with col2:
    st.subheader("Steam Loss vs Orifice Diameter Graph")
    try:
        st.image("image_3ea5b3.jpg", caption="Steam Loss (kg/hr) at varying pressures and diameters")
    except Exception as e:
        st.warning("Please upload 'image_3ea5b3.jpg' to the working directory.")

st.markdown("""
### Nomenclature
* **W**: Steam loss based on basic Napier's empirical formula (lb/hr)
* **L**: Actual loss of steam across the system (kg/hr)
* **D / d**: Diameter of the sharp-edged orifice (inches / mm)
* **P_in**: Pressure of steam at inlet of trap (psi / barg)
* **P_out**: Pressure of steam at outlet of trap (psi / barg)
* **F_T**: Failure factor of trap (empirical, default = 1)
* **F_S**: Service factor of trap 
* **C_V**: Flow coefficient of the trap
* **h**: Operating hours of the trap
""")

st.latex(r"L = F_T \times F_S \times C_V \times h \times \sqrt{(P_{in} - P_{out})(P_{in} + P_{out})}")

# -----------------------------------------------------------------------------
# 3. Interactive Inputs
# -----------------------------------------------------------------------------
st.header("2. Input Parameters")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Select Trap Specifications")
    use_custom_orifice = st.checkbox("Enter Custom Orifice Size (Override Catalog)")
    
    if not use_custom_orifice:
        product = st.selectbox("Select Product Name", df['Product Name'].unique())
        filtered_df = df[df['Product Name'] == product]
        
        size_dn = st.selectbox("Select Size (DN)", filtered_df['Size (DN)'].unique())
        filtered_df = filtered_df[filtered_df['Size (DN)'] == size_dn]
        
        dp = st.selectbox("Select Differential Pressure (bar g)", filtered_df['Differential Pressure (bar g)'].unique())
        
        # Get corresponding orifice size
        orifice_mm_str = filtered_df[filtered_df['Differential Pressure (bar g)'] == dp]['Orifice Size (mm)'].values[0]
        try:
            orifice_mm = float(orifice_mm_str)
        except:
            st.warning("Orifice size for this selection is missing or non-numeric. Please use custom override.")
            orifice_mm = 0.0
            
        st.success(f"Selected Orifice Size: {orifice_mm} mm")
    else:
        orifice_mm = st.number_input("Custom Orifice Size (mm)", min_value=0.1, value=4.0, step=0.1)

with col_b:
    st.subheader("Operating Conditions")
    p_in_barg = st.number_input("Inlet Pressure (barg)", min_value=0.0, value=10.0, step=0.5)
    p_out_barg = st.number_input("Outlet Pressure (barg)", min_value=0.0, value=2.0, step=0.5)
    hours = st.number_input("Operating Hours per Year", min_value=1, max_value=8760, value=8000, step=100)
    application = st.radio("Application Type", ["Steam Mains & Headers", "Process"])

# -----------------------------------------------------------------------------
# 4. Mathematical Conversions & Calculations
# -----------------------------------------------------------------------------
# Conversions
d_inches = orifice_mm / 25.4
p_in_psi = p_in_barg * 14.5038
p_out_psi = p_out_barg * 14.5038

# Constants
F_T = 1.0
C_V = 22.1 * (d_inches ** 2)

# Logic Branches
loss_kg_hr = 0.0

if application == "Steam Mains & Headers":
    F_S = 1.4
    if p_in_psi >= p_out_psi:
        loss_kg_hr = 1.4 * C_V * np.sqrt((p_in_psi - p_out_psi) * (p_in_psi + p_out_psi))
    else:
        st.error("Outlet pressure cannot be greater than inlet pressure.")

elif application == "Process":
    F_S = 0.9
    # Check condition A vs B
    if p_out_psi >= (p_in_psi / 2):
        loss_kg_hr = 0.25 * C_V * np.sqrt((p_in_psi - p_out_psi) * (p_in_psi + p_out_psi))
    else:
        loss_kg_hr = 0.25 * C_V * np.sqrt((p_in_psi - p_in_psi/2) * (p_in_psi + p_in_psi/2))

total_loss_yr = loss_kg_hr * hours

# -----------------------------------------------------------------------------
# 5. Output Results
# -----------------------------------------------------------------------------
st.header("3. Calculation Results")
st.markdown("---")

col_res1, col_res2, col_res3 = st.columns(3)

col_res1.metric("Converted Orifice Diameter", f"{d_inches:.4f} in")
col_res1.metric("Converted Inlet Pressure", f"{p_in_psi:.2f} psi")
col_res1.metric("Converted Outlet Pressure", f"{p_out_psi:.2f} psi")

col_res2.metric("Flow Coefficient (C_V)", f"{C_V:.4f}")
col_res2.metric("Service Factor (F_S)", f"{F_S}")
col_res2.metric("Failure Factor (F_T)", f"{F_T}")

col_res3.metric("Steam Loss Rate", f"{loss_kg_hr:.2f} kg/hr")
col_res3.metric("Total Annual Loss", f"{total_loss_yr:,.2f} kg/yr")

st.markdown("---")
st.info("Note: The calculations utilize the generalized industrial adaptation of Napier's formula, converting inputs from Metric (barg, mm) to Imperial (psi, inches) internally before executing the logic tree based on the application type.")
