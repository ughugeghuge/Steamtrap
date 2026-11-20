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
# Clean up missing Orifice sizes for filtering
df['Orifice Size (mm)'] = pd.to_numeric(df['Orifice Size (mm)'], errors='coerce')
df = df.dropna(subset=['Orifice Size (mm)'])

# -----------------------------------------------------------------------------
# 2. Calculation Logic & Functions
# -----------------------------------------------------------------------------
def calc_service_factor(S):
    """Calculate Service Factor (F_S) from Capacity Factor (S)"""
    if S == 0: return 0
    return 2.1 * (S - 1) / S

def calc_steam_loss(d_mm, p_in_barg, p_out_barg, application):
    """Calculates steam loss (kg/hr) strictly based on document logic."""
    if d_mm <= 0:
        return 0.0
        
    d_in = d_mm / 25.4
    p_in_psi = p_in_barg * 14.5038
    p_out_psi = p_out_barg * 14.5038
    
    cv = 22.1 * (d_in ** 2)
    
    if application == "Steam Mains & Headers":
        if p_out_psi >= p_in_psi:
            return 0.0
        # Steam Mains Formula: Loss = 1.4 * 22.1 * d^2 * sqrt(...)
        return 1.4 * cv * np.sqrt((p_in_psi - p_out_psi) * (p_in_psi + p_out_psi))
        
    elif application == "Process":
        if p_out_psi >= (p_in_psi / 2):
            # Condition A
            return 0.25 * cv * np.sqrt((p_in_psi - p_out_psi) * (p_in_psi + p_out_psi))
        else:
            # Condition B (P_out < P_in / 2)
            return 0.25 * cv * np.sqrt((p_in_psi - p_in_psi/2) * (p_in_psi + p_in_psi/2))
    
    return 0.0

def draw_orifice_schematic():
    """Generates a dynamic matplotlib schematic of a sharp-edged orifice."""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot([0, 10], [3, 3], 'k-', lw=3)
    ax.plot([0, 10], [1, 1], 'k-', lw=3)
    ax.add_patch(patches.Rectangle((4.8, 2.3), 0.4, 0.7, facecolor='gray', edgecolor='black'))
    ax.add_patch(patches.Rectangle((4.8, 1.0), 0.4, 0.7, facecolor='gray', edgecolor='black'))
    
    ax.arrow(1, 2, 2, 0, head_width=0.2, head_length=0.3, fc='blue', ec='blue')
    ax.arrow(1.5, 2.5, 2, 0, head_width=0.15, head_length=0.2, fc='lightblue', ec='lightblue')
    ax.arrow(1.5, 1.5, 2, 0, head_width=0.15, head_length=0.2, fc='lightblue', ec='lightblue')
    ax.arrow(5.5, 2, 2, 0, head_width=0.2, head_length=0.3, fc='blue', ec='blue')
    
    ax.text(2, 3.2, 'Inlet Pressure ($P_{in}$)', fontsize=12, color='darkred', fontweight='bold', ha='center')
    ax.text(8, 3.2, 'Outlet Pressure ($P_{out}$)', fontsize=12, color='darkgreen', fontweight='bold', ha='center')
    ax.text(5, 0.6, 'Orifice Diameter ($D$)', fontsize=12, color='black', fontweight='bold', ha='center')
    ax.plot([5, 5], [1.7, 2.3], 'r--', lw=2)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    return fig

# -----------------------------------------------------------------------------
# 3. Main Streamlit App
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Steam Trap Loss Calculator", layout="wide")
st.title("Steam Trap Loss Calculator & Modeler")

# Setup Tabs
tab_theory, tab_inputs, tab_calcs, tab_visuals = st.tabs([
    "📖 Theory & Nomenclature", 
    "⚙️ Inputs & Conversions", 
    "🧮 Calculations", 
    "📈 Graph Visuals"
])

# ==========================================
# TAB 1: THEORY & NOMENCLATURE
# ==========================================
with tab_theory:
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
        * **S**: Capacity Factor (Condensate load on trap)
        * **F_S**: Service factor of trap (Derived via $F_S = 2.1 \\times (S - 1) / S$)
        * **C_V**: Flow coefficient of the trap ($C_V = 22.1 \\times D^2$)
        * **h**: Operating hours of the trap
        """)
        st.latex(r"L = F_T \times F_S \times C_V \times h \times \sqrt{(P_{in} - P_{out})(P_{in} + P_{out})}")
        
        st.markdown("**Strict Process Condition Logic:**")
        st.markdown("- If $P_{out} \\ge P_{in}/2$: Uses actual $P_{out}$ in formula.")
        st.markdown("- If $P_{out} < P_{in}/2$: Caps differential using $P_{in}/2$ in formula.")

# ==========================================
# TAB 2: INPUTS & CONVERSIONS
# ==========================================
with tab_inputs:
    col_in, col_conv = st.columns([1.2, 1])
    
    with col_in:
        st.subheader("1. Operating Conditions")
        p_in_barg = st.number_input("Inlet Pressure (barg)", min_value=0.0, value=10.0, step=0.5)
        p_out_barg = st.number_input("Outlet Pressure (barg)", min_value=0.0, value=2.0, step=0.5)
        hours = st.number_input("Operating Hours per Year", min_value=1, max_value=8760, value=8000, step=100)
        
        st.markdown("---")
        st.subheader("2. Application Setup")
        application = st.radio("Application Type", ["Steam Mains & Headers", "Process"])
        
        if application == "Steam Mains & Headers":
            s_factor = st.number_input("Capacity Factor (S)", value=3.0)
            f_s = calc_service_factor(s_factor)
        else:
            s_factor = st.number_input("Capacity Factor (S)", value=1.75)
            f_s = calc_service_factor(s_factor)
            
        st.markdown("---")
        st.subheader("3. Select Trap Models (Multi-Select)")
        selected_products = st.multiselect("Select Product(s)", df['Product Name'].unique(), default=["SOFT31"])
        
        filtered_df = df[df['Product Name'].isin(selected_products)]
        
        if not filtered_df.empty:
            selected_sizes = st.multiselect("Filter by Size (DN)", filtered_df['Size (DN)'].unique(), default=filtered_df['Size (DN)'].unique())
            filtered_df = filtered_df[filtered_df['Size (DN)'].isin(selected_sizes)]
            
            selected_dp = st.multiselect("Filter by Differential Pressure (bar g)", filtered_df['Differential Pressure (bar g)'].unique(), default=filtered_df['Differential Pressure (bar g)'].unique())
            filtered_df = filtered_df[filtered_df['Differential Pressure (bar g)'].isin(selected_dp)]
            
        st.write("### Selected Traps Data", filtered_df)

    with col_conv:
        st.subheader("Live Converted Imperial Values")
        st.info("The math model uses Imperial standard logic natively.")
        
        p_in_psi = p_in_barg * 14.5038
        p_out_psi = p_out_barg * 14.5038
        
        st.metric("Inlet Pressure ($P_{in}$)", f"{p_in_psi:.2f} psi")
        st.metric("Outlet Pressure ($P_{out}$)", f"{p_out_psi:.2f} psi")
        st.metric("Capacity Factor ($S$)", f"{s_factor}")
        st.metric("Service Factor ($F_S$)", f"{f_s:.3f}")
        
        # Display logic check
        if application == "Process":
            if p_out_psi >= (p_in_psi / 2):
                st.warning(f"Condition A: $P_{{out}}$ ({p_out_psi:.1f}) >= $P_{{in}}/2$ ({p_in_psi/2:.1f}). Using actual $P_{{out}}$.")
            else:
                st.error(f"Condition B: $P_{{out}}$ ({p_out_psi:.1f}) < $P_{{in}}/2$ ({p_in_psi/2:.1f}). Equation will use $P_{{in}}/2$.")

# ==========================================
# TAB 3: CALCULATIONS
# ==========================================
with tab_calcs:
    st.subheader("Computed Steam Loss for Selected Traps")
    
    if not filtered_df.empty:
        results = []
        for index, row in filtered_df.iterrows():
            d_mm = row['Orifice Size (mm)']
            d_in = d_mm / 25.4
            cv = 22.1 * (d_in ** 2)
            
            loss_hr = calc_steam_loss(d_mm, p_in_barg, p_out_barg, application)
            loss_yr = loss_hr * hours
            
            results.append({
                "Product": row['Product Name'],
                "Size (DN)": row['Size (DN)'],
                "DP (bar g)": row['Differential Pressure (bar g)'],
                "Orifice Dia (mm)": d_mm,
                "Flow Coeff (C_V)": round(cv, 4),
                "Loss (kg/hr)": round(loss_hr, 2),
                "Annual Loss (kg/yr)": round(loss_yr, 2)
            })
            
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
    else:
        st.warning("No traps selected to calculate.")

# ==========================================
# TAB 4: GRAPH VISUALS
# ==========================================
with tab_visuals:
    st.subheader("Dynamic Multi-Trap Sensitivity Analysis")
    st.markdown("Plots steam loss curves if orifice size wears/varies. Dots indicate standard catalog operating points.")
    
    if not filtered_df.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # We don't want duplicate curves if they have the same orifice size. Group by orifice.
        unique_traps = filtered_df.drop_duplicates(subset=['Product Name', 'Orifice Size (mm)'])
        
        for index, row in unique_traps.iterrows():
            d_mm = row['Orifice Size (mm)']
            name = f"{row['Product Name']} (DN{row['Size (DN)']}, {row['Differential Pressure (bar g)']}bar)"
            
            # Generate array of orifice sizes from 1mm up to d_mm * 2
            dia_range = np.linspace(1.0, max(10.0, d_mm * 2.0), 100)
            loss_range = [calc_steam_loss(d, p_in_barg, p_out_barg, application) for d in dia_range]
            
            # Plot Curve
            line, = ax2.plot(dia_range, loss_range, linewidth=2, label=f"Curve: {name}")
            
            # Plot specific operating point for this trap
            actual_loss = calc_steam_loss(d_mm, p_in_barg, p_out_barg, application)
            ax2.plot(d_mm, actual_loss, marker='o', markersize=8, color=line.get_color())
            
            # Add text annotation near the dot
            ax2.text(d_mm, actual_loss, f" {d_mm}mm", fontsize=9, verticalalignment='bottom')

        ax2.set_xlabel("Orifice Diameter (mm)", fontweight='bold')
        ax2.set_ylabel("Steam Loss (kg/hr)", fontweight='bold')
        ax2.set_title(f"Impact of Orifice Variation on Steam Loss ({application}) @ {p_in_barg} barg Inlet", fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend(loc="upper left", bbox_to_anchor=(1, 1)) # Put legend outside the plot
        
        st.pyplot(fig2)
    else:
        st.warning("Select traps in the 'Inputs' tab to generate visualizations.")
