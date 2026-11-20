import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io

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
# 2. Calculation Logic Module
# -----------------------------------------------------------------------------
def calc_steam_loss(d_mm, p_in_barg, p_out_barg, application):
    """Calculates steam loss (kg/hr) based on generalized Napier's formula."""
    if d_mm <= 0:
        return 0.0
        
    d_in = d_mm / 25.4
    p_in_psi = p_in_barg * 14.5038
    p_out_psi = p_out_barg * 14.5038
    
    cv = 22.1 * (d_in ** 2)
    
    if application == "Steam Mains & Headers":
        if p_out_psi > p_in_psi:
            return 0.0
        return 1.4 * cv * np.sqrt((p_in_psi - p_out_psi) * (p_in_psi + p_out_psi))
        
    elif application == "Process":
        if p_out_psi >= (p_in_psi / 2):
            return 0.25 * cv * np.sqrt((p_in_psi - p_out_psi) * (p_in_psi + p_out_psi))
        else:
            return 0.25 * cv * np.sqrt((p_in_psi - p_in_psi/2) * (p_in_psi + p_in_psi/2))
    
    return 0.0

# -----------------------------------------------------------------------------
# 3. Drawing Generators
# -----------------------------------------------------------------------------
def draw_orifice_schematic():
    """Generates a dynamic matplotlib schematic of a sharp-edged orifice."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Draw pipe walls
    ax.plot([0, 10], [3, 3], 'k-', lw=3)
    ax.plot([0, 10], [1, 1], 'k-', lw=3)
    
    # Draw sharp edged orifice
    ax.add_patch(patches.Rectangle((4.8, 2.3), 0.4, 0.7, facecolor='gray', edgecolor='black'))
    ax.add_patch(patches.Rectangle((4.8, 1.0), 0.4, 0.7, facecolor='gray', edgecolor='black'))
    
    # Flow Arrows
    ax.arrow(1, 2, 2, 0, head_width=0.2, head_length=0.3, fc='blue', ec='blue')
    ax.arrow(1.5, 2.5, 2, 0, head_width=0.15, head_length=0.2, fc='lightblue', ec='lightblue')
    ax.arrow(1.5, 1.5, 2, 0, head_width=0.15, head_length=0.2, fc='lightblue', ec='lightblue')
    
    # Choked flow arrows
    ax.arrow(5.5, 2, 2, 0, head_width=0.2, head_length=0.3, fc='blue', ec='blue')
    
    # Labels
    ax.text(2, 3.2, 'Inlet Pressure ($P_{in}$)', fontsize=12, color='darkred', fontweight='bold', ha='center')
    ax.text(8, 3.2, 'Outlet Pressure ($P_{out}$)', fontsize=12, color='darkgreen', fontweight='bold', ha='center')
    ax.text(5, 0.6, 'Orifice Diameter ($D$)', fontsize=12, color='black', fontweight='bold', ha='center')
    ax.text(2, 0.6, 'Saturated Steam Flow', fontsize=10, style='italic', ha='center')
    
    # Orifice Dotted Line
    ax.plot([5, 5], [1.7, 2.3], 'r--', lw=2)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    return fig

# -----------------------------------------------------------------------------
# 4. UI Setup & Theory
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Steam Trap Loss Calculator", layout="wide")
st.title("Steam Trap Loss Calculator & Modeler")

st.header("1. Theory & Nomenclature")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Napier's Equation & Orifice Construction")
    st.pyplot(draw_orifice_schematic())
    
with col2:
    st.subheader("Nomenclature")
    st.markdown("""
    * **W**: Steam loss based on basic Napier's empirical formula (lb/hr)
    * **L**: Actual loss of steam across the system (kg/hr)
    * **D / d**: Diameter of the sharp-edged orifice (inches / mm)
    * **P_in**: Pressure of steam at inlet of trap (psi / barg)
    * **P_out**: Pressure of steam at outlet of trap (psi / barg)
    * **F_T**: Failure factor of trap (empirical, default = 1)
    * **F_S**: Service factor of trap (Mains = 1.4, Process = 0.9)
    * **C_V**: Flow coefficient of the trap (22.1 * D²)
    * **h**: Operating hours of the trap
    """)
    st.latex(r"L = F_T \times F_S \times C_V \times h \times \sqrt{(P_{in} - P_{out})(P_{in} + P_{out})}")

# -----------------------------------------------------------------------------
# 5. Interactive Inputs
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
        
        orifice_mm_str = filtered_df[filtered_df['Differential Pressure (bar g)'] == dp]['Orifice Size (mm)'].values[0]
        try:
            orifice_mm = float(orifice_mm_str)
        except:
            st.warning("Orifice size for this selection is missing or non-numeric. Defaulting to 4.0 mm.")
            orifice_mm = 4.0
            
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
# 6. Perform Calculations
# -----------------------------------------------------------------------------
# Current operating point calculation
loss_kg_hr = calc_steam_loss(orifice_mm, p_in_barg, p_out_barg, application)
total_loss_yr = loss_kg_hr * hours

d_inches = orifice_mm / 25.4
p_in_psi = p_in_barg * 14.5038
p_out_psi = p_out_barg * 14.5038
C_V = 22.1 * (d_inches ** 2)
F_S = 1.4 if application == "Steam Mains & Headers" else 0.9

# -----------------------------------------------------------------------------
# 7. Output Results & Dynamic Chart
# -----------------------------------------------------------------------------
st.header("3. Calculation Results & Sensitivity Analysis")
st.markdown("---")

col_res1, col_res2, col_res3 = st.columns(3)

col_res1.metric("Converted Orifice Dia.", f"{d_inches:.4f} in")
col_res1.metric("Converted Inlet Press.", f"{p_in_psi:.2f} psi")
col_res1.metric("Converted Outlet Press.", f"{p_out_psi:.2f} psi")

col_res2.metric("Flow Coefficient (C_V)", f"{C_V:.4f}")
col_res2.metric("Service Factor (F_S)", f"{F_S}")
col_res2.metric("Failure Factor (F_T)", "1.0")

col_res3.metric("Steam Loss Rate", f"{loss_kg_hr:.2f} kg/hr")
col_res3.metric("Total Annual Loss", f"{total_loss_yr:,.2f} kg/yr")

st.markdown("---")

# --- Dynamic Chart Generation ---
st.subheader("Dynamic Sensitivity: Steam Loss vs Orifice Diameter")

# Generate an array of orifice sizes from 20% to 200% of selected size
dia_range = np.linspace(max(0.5, orifice_mm * 0.2), orifice_mm * 2.0, 100)
loss_range = [calc_steam_loss(d, p_in_barg, p_out_barg, application) for d in dia_range]

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(dia_range, loss_range, label=f'Loss Curve @ {p_in_barg} barg Inlet', color='navy', linewidth=2)
ax2.plot(orifice_mm, loss_kg_hr, marker='o', markersize=10, color='red', label=f'Current Op Point ({orifice_mm}mm, {loss_kg_hr:.1f} kg/hr)')

ax2.set_xlabel("Orifice Diameter (mm)", fontweight='bold')
ax2.set_ylabel("Steam Loss (kg/hr)", fontweight='bold')
ax2.set_title(f"Impact of Orifice Wear on Steam Loss ({application})", fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.7)
ax2.legend()

st.pyplot(fig2)

st.info("Note: The calculations utilize the generalized industrial adaptation of Napier's formula, converting inputs from Metric (barg, mm) to Imperial (psi, inches) internally before executing the logic tree based on the application type.")
