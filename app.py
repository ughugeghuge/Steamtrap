import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
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
df['Orifice Size (mm)'] = pd.to_numeric(df['Orifice Size (mm)'], errors='coerce')
df = df.dropna(subset=['Orifice Size (mm)'])

# -----------------------------------------------------------------------------
# 2. Mathematical Functions
# -----------------------------------------------------------------------------
def calc_service_factor(S):
    if S <= 0: return 0
    return 2.1 * (S - 1) / S

def calc_steam_loss(d_mm, p_in_barg, p_out_barg, application, f_t=1.0):
    """Calculates steam loss natively in kg/hr using absolute pressure."""
    if d_mm <= 0: return 0.0
    d_in = d_mm / 25.4
    
    # CRITICAL UNIT FIX: Convert barg to Absolute Pressure (psia) for Napier's equation
    p_in_psia = (p_in_barg * 14.5038) + 14.696
    p_out_psia = (p_out_barg * 14.5038) + 14.696
    
    cv = 22.1 * (d_in ** 2)
    
    if application == "Steam Mains & Headers":
        if p_out_psia >= p_in_psia: return 0.0
        return f_t * 1.4 * cv * np.sqrt((p_in_psia - p_out_psia) * (p_in_psia + p_out_psia))
        
    elif application == "Process":
        if p_out_psia >= (p_in_psia / 2):
            return f_t * 0.25 * cv * np.sqrt((p_in_psia - p_out_psia) * (p_in_psia + p_out_psia))
        else:
            return f_t * 0.25 * cv * np.sqrt((p_in_psia - p_in_psia/2) * (p_in_psia + p_in_psia/2))
    return 0.0

# -----------------------------------------------------------------------------
# 3. Chart Design Functions (Static Matplotlib Schematic)
# -----------------------------------------------------------------------------
def draw_orifice_schematic():
    fig, ax = plt.subplots(figsize=(7, 3), facecolor='white')
    ax.set_facecolor('white')
    
    ax.plot([0, 10], [3, 3], 'k-', lw=2.5)
    ax.plot([0, 10], [1, 1], 'k-', lw=2.5)
    ax.add_patch(patches.Rectangle((4.8, 2.3), 0.4, 0.7, facecolor='#7f8c8d', edgecolor='none'))
    ax.add_patch(patches.Rectangle((4.8, 1.0), 0.4, 0.7, facecolor='#7f8c8d', edgecolor='none'))
    
    ax.arrow(1, 2, 2, 0, head_width=0.2, head_length=0.3, fc='#3498db', ec='#3498db')
    ax.arrow(1.5, 2.5, 2, 0, head_width=0.15, head_length=0.2, fc='#AED6F1', ec='#AED6F1')
    ax.arrow(1.5, 1.5, 2, 0, head_width=0.15, head_length=0.2, fc='#AED6F1', ec='#AED6F1')
    ax.arrow(5.5, 2, 2, 0, head_width=0.2, head_length=0.3, fc='#3498db', ec='#3498db')
    
    ax.text(2, 3.2, 'Inlet Pressure (Pin, psia)', fontsize=11, color='#c0392b', fontweight='bold', ha='center')
    ax.text(8, 3.2, 'Outlet Pressure (Pout, psia)', fontsize=11, color='#27ae60', fontweight='bold', ha='center')
    ax.text(5, 0.6, 'Orifice Diameter (D)', fontsize=11, color='#2c3e50', fontweight='bold', ha='center')
    ax.plot([5, 5], [1.7, 2.3], 'r--', lw=1.5)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    plt.tight_layout()
    return fig

# Mapping dictionaries
fail_options = {
    "Blowthrough (BT)": 1.0,
    "Leaking (LK)": 0.25,
    "Rapid Cycle (RC)": 0.2
}
fail_plotly_styles = {
    "Blowthrough (BT)": "solid",
    "Leaking (LK)": "dash",
    "Rapid Cycle (RC)": "dot"
}
theme_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']

# -----------------------------------------------------------------------------
# 4. Main Streamlit Interface
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Steam Trap Loss Calculator", layout="wide")
st.title("Steam Trap Loss Modeler")
st.markdown("<style> .stTabs [data-baseweb='tab-list'] { gap: 20px; } </style>", unsafe_allow_html=True)

tab_theory, tab_inputs, tab_calcs, tab_visuals = st.tabs([
    "Theory & Nomenclature", 
    "Inputs & Conversions", 
    "Calculations", 
    "Interactive Visuals"
])

# --- TAB 1: THEORY & NOMENCLATURE ---
with tab_theory:
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("Orifice Construction")
        st.pyplot(draw_orifice_schematic())
        st.subheader("General Equation")
        st.latex(r"L = F_T \times F_S \times C_V \times h \times \sqrt{(P_{in} - P_{out})(P_{in} + P_{out})}")
    with col2:
        st.subheader("Nomenclature")
        st.markdown("""
        * **L**: Actual loss of steam across the system (kg/hr)
        * **d**: Diameter of the sharp-edged orifice (inches)
        * **Pin**: Pressure of steam at inlet of trap (Absolute Pressure: psia)
        * **Pout**: Pressure of steam at outlet of trap (Absolute Pressure: psia)
        * **h**: Operating hours of the trap
        """)
        st.subheader("System Factors")
        st.markdown("**Failure Factor (FT)** is based on the trap's condition:")
        st.markdown("- **Blowthrough (BT)**: Trap blowing full bore ($F_T = 1.0$)")
        st.markdown("- **Leaking (LK)**: Partial leak ($F_T = 0.25$)")
        st.markdown("- **Rapid Cycle (RC)**: Motor boating / cycling ($F_T = 0.2$)")
        st.markdown("**Service Factor (FS)**:")
        st.latex(r"F_S = \frac{2.1 \times (S - 1)}{S}")
        st.markdown("- **Steam Mains**: Capacity Factor (S) = 3.0, resulting in FS = 1.4")
        st.markdown("- **Process**: Capacity Factor (S) = 1.75, resulting in FS = 0.9")
        st.markdown("**Flow Coefficient (CV)**: $C_V = 22.1 \times d^2$")
    st.markdown("---")
    st.subheader("Application Specific Formulas (Rates in kg/hr)")
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("**1. Steam Traps Draining Steam Mains and Headers**")
        st.latex(r"Loss = F_T \times 1.4 \times 22.1 \times d^2 \times \sqrt{(P_{in} - P_{out})(P_{in} + P_{out})}")
    with col4:
        st.markdown("**2. Process Steam Traps**")
        st.markdown("Condition A: $P_{out} \ge P_{in}/2$")
        st.latex(r"Loss = F_T \times 0.25 \times 22.1 \times d^2 \times \sqrt{(P_{in} - P_{out})(P_{in} + P_{out})}")
        st.markdown("Condition B: $P_{out} < P_{in}/2$")
        st.latex(r"Loss = F_T \times 0.25 \times 22.1 \times d^2 \times \sqrt{(P_{in} - P_{in}/2)(P_{in} + P_{in}/2)}")

# --- TAB 2: INPUTS & CONVERSIONS ---
with tab_inputs:
    col_in, col_conv = st.columns([1.3, 1])
    
    with col_in:
        st.subheader("Operating Conditions")
        c1, c2, c_hrs = st.columns(3)
        p_in_barg = c1.number_input("Inlet Pressure (barg)", min_value=0.0, value=10.0, step=0.5, key='in_pin')
        p_out_barg = c2.number_input("Outlet Pressure (barg)", min_value=0.0, value=2.0, step=0.5, key='in_pout')
        hours = c_hrs.number_input("Operating Hours per Year", min_value=1, max_value=8760, value=8000, step=100)
        
        st.markdown("---")
        st.subheader("Application & Failure Setup")
        c_app, c_fail = st.columns(2)
        application = c_app.radio("Application Type", ["Steam Mains & Headers", "Process"], key='in_app')
        failure_type = c_fail.radio("Current Failure Status", list(fail_options.keys()))
        
        s_factor = st.number_input("Capacity Factor (S)", value=3.0 if application == "Steam Mains & Headers" else 1.75)
        f_s = calc_service_factor(s_factor)
        f_t = fail_options[failure_type]
            
        st.markdown("---")
        st.subheader("Trap Selection")
        selected_products = st.multiselect("Select Product(s)", df['Product Name'].unique(), default=["SOFT31"])
        filtered_df = df[df['Product Name'].isin(selected_products)]
        
        if not filtered_df.empty:
            c3, c4 = st.columns(2)
            selected_sizes = c3.multiselect("Filter by Size (DN)", filtered_df['Size (DN)'].unique(), default=filtered_df['Size (DN)'].unique())
            filtered_df = filtered_df[filtered_df['Size (DN)'].isin(selected_sizes)]
            selected_dp = c4.multiselect("Filter by Differential Pressure (bar g)", filtered_df['Differential Pressure (bar g)'].unique(), default=filtered_df['Differential Pressure (bar g)'].unique())
            filtered_df = filtered_df[filtered_df['Differential Pressure (bar g)'].isin(selected_dp)]
            
        st.write("Selected Configuration Matrix", filtered_df)

    with col_conv:
        st.subheader("Converted Absolute Parameters")
        st.info("Napier's logic strictly requires Absolute Pressure (psia) by adding atmospheric pressure (14.696 psi) to the converted gauge pressure.")
        p_in_psia = (p_in_barg * 14.5038) + 14.696
        p_out_psia = (p_out_barg * 14.5038) + 14.696
        st.metric("Inlet Pressure (Pin)", f"{p_in_psia:.2f} psia")
        st.metric("Outlet Pressure (Pout)", f"{p_out_psia:.2f} psia")
        st.metric("Service Factor (Fs)", f"{f_s:.3f}")
        st.metric("Failure Factor (FT)", f"{f_t}")
        
        if application == "Process":
            if p_out_psia >= (p_in_psia / 2):
                st.info(f"Condition A Applied: Pout ({p_out_psia:.1f}) >= Pin/2 ({p_in_psia/2:.1f})")
            else:
                st.info(f"Condition B Applied: Pout ({p_out_psia:.1f}) < Pin/2 ({p_in_psia/2:.1f})")

# --- TAB 3: CALCULATIONS ---
with tab_calcs:
    st.subheader("Computed Steam Loss Data")
    if not filtered_df.empty:
        results = []
        for index, row in filtered_df.iterrows():
            d_mm = row['Orifice Size (mm)']
            d_in = d_mm / 25.4
            cv = 22.1 * (d_in ** 2)
            loss_hr = calc_steam_loss(d_mm, p_in_barg, p_out_barg, application, f_t)
            loss_yr = loss_hr * hours
            results.append({
                "Product": row['Product Name'],
                "Size (DN)": row['Size (DN)'],
                "DP (bar g)": row['Differential Pressure (bar g)'],
                "Orifice (mm)": d_mm,
                "Flow Coeff (Cv)": round(cv, 4),
                "Loss (kg/hr)": round(loss_hr, 2),
                "Annual Loss (kg/yr)": round(loss_yr, 2)
            })
        st.dataframe(pd.DataFrame(results), use_container_width=True)
    else:
        st.write("No traps selected for calculation.")

# --- TAB 4: INTERACTIVE VISUALS ---
with tab_visuals:
    st.subheader("Standalone Interactive Dashboard")
    st.markdown("This dashboard is completely isolated from the inputs tab. Use the interactive legends on the right side of each graph to click and toggle specific lines on or off.")
    
    unique_traps_all = df.drop_duplicates(subset=['Product Name', 'Size (DN)', 'Orifice Size (mm)'])
    trap_options_all = {f"{row['Product Name']} - DN{row['Size (DN)']} (Orifice: {row['Orifice Size (mm)']}mm, DP: {row['Differential Pressure (bar g)']}bar)": row for index, row in unique_traps_all.iterrows()}
        
    with st.container():
        st.markdown("#### Independent Visualization Controls")
        row1_col1, row1_col2 = st.columns([1, 1])
        with row1_col1:
            viz_trap_name = st.selectbox("Select Any Trap to Visualize", list(trap_options_all.keys()))
            viz_app = st.radio("Application Setup", ["Steam Mains & Headers", "Process"])
        with row1_col2:
            viz_p_out = st.slider("Select Back Pressure (Pout) for Wear Analysis (barg)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
            viz_p_in_op = st.number_input("Operating Inlet Pressure (Pin) to mark on wear graph", min_value=0.0, value=10.0, step=0.5)
        
        row2_col1, row2_col2 = st.columns([1, 1])
        with row2_col1:
            viz_pins = st.multiselect("Standard Inlet Pressures (barg)", [4.5, 10.0, 13.0, 14.0, 21.0], default=[4.5, 10.0, 13.0, 14.0, 21.0])
        with row2_col2:
            viz_custom_pin = st.number_input("Custom Inlet Pressure (barg)", min_value=0.0, value=0.0, step=0.5)
            
        viz_fails = st.multiselect("Select Failure Types to Overlay", list(fail_options.keys()), default=list(fail_options.keys()))

        with st.expander("Axis Scaling Controls (Optional - Graphs are fully interactive)"):
            st.markdown("Use these inputs to strictly clamp the axis boundaries if you do not want to use your mouse to zoom/pan.")
            g1c1, g1c2, g1c3, g1c4 = st.columns(4)
            g1_x_max = g1c2.number_input("Graph 1 X Max (barg)", value=10.0, step=1.0)
            g1_y_max = g1c4.number_input("Graph 1 Y Max (kg/hr)", value=250.0, step=50.0)
            
            g2c1, g2c2, g2c3, g2c4 = st.columns(4)
            g2_x_max = g2c2.number_input("Graph 2 X Max (mm)", value=15.0, step=1.0)
            g2_y_max = g2c4.number_input("Graph 2 Y Max (kg/hr)", value=1500.0, step=100.0)

    plot_pins = list(viz_pins)
    if viz_custom_pin > 0 and viz_custom_pin not in plot_pins:
        plot_pins.append(viz_custom_pin)
    plot_pins.sort()
        
    viz_trap_data = trap_options_all[viz_trap_name]
    d_mm = viz_trap_data['Orifice Size (mm)']
    
    st.markdown("---")
    
    # --- Interactive Plotly Graph 1 ---
    fig1 = go.Figure()
    p_out_range = np.linspace(0, g1_x_max, 100)
    
    if plot_pins and viz_fails:
        for i, pin in enumerate(plot_pins):
            c = theme_colors[i % len(theme_colors)]
            for f_name in viz_fails:
                ft_val = fail_options[f_name]
                ls = fail_plotly_styles[f_name]
                label_str = f'Pin={pin} | {f_name.split()[0]}' if len(viz_fails) > 1 else f'Pin={pin} barg'
                loss_range_pout = [calc_steam_loss(d_mm, pin, pout, viz_app, ft_val) for pout in p_out_range]
                
                fig1.add_trace(go.Scatter(x=p_out_range, y=loss_range_pout, mode='lines', name=label_str, line=dict(color=c, dash=ls, width=2.5)))
    
    fig1.add_vline(x=viz_p_out, line_width=1.5, line_dash="dash", line_color="gray", annotation_text=f" Selected Pout ({viz_p_out})")
    
    fig1.update_layout(
        title=f"Loss vs. Back Pressure (Fixed Orifice: {d_mm}mm)",
        xaxis_title="Back Pressure Pout (barg)",
        yaxis_title="Steam Loss (kg/hr)",
        xaxis=dict(range=[0, g1_x_max]),
        yaxis=dict(range=[0, g1_y_max]),
        hovermode="x unified",
        template="simple_white",
        height=600,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --- Interactive Plotly Graph 2 ---
    fig2 = go.Figure()
    dia_range = np.linspace(1.0, g2_x_max, 100)
    
    if plot_pins and viz_fails:
        for i, pin in enumerate(plot_pins):
            c = theme_colors[i % len(theme_colors)]
            for f_name in viz_fails:
                ft_val = fail_options[f_name]
                ls = fail_plotly_styles[f_name]
                label_str = f'Pin={pin} | {f_name.split()[0]}' if len(viz_fails) > 1 else f'Pin={pin} barg'
                loss_range_dia = [calc_steam_loss(d, pin, viz_p_out, viz_app, ft_val) for d in dia_range]
                
                fig2.add_trace(go.Scatter(x=dia_range, y=loss_range_dia, mode='lines', name=label_str, line=dict(color=c, dash=ls, width=2.5)))
    
    current_loss_viz = calc_steam_loss(d_mm, viz_p_in_op, viz_p_out, viz_app, fail_options["Blowthrough (BT)"])
    fig2.add_trace(go.Scatter(x=[d_mm], y=[current_loss_viz], mode='markers', name=f'Op Point (@ {viz_p_in_op} barg, BT)', marker=dict(color='black', size=14, symbol='star')))

    fig2.update_layout(
        title=f"Loss vs. Orifice Wear (Fixed Pout: {viz_p_out} barg)",
        xaxis_title="Orifice Diameter (mm)",
        yaxis_title="Steam Loss (kg/hr)",
        xaxis=dict(range=[0, g2_x_max]),
        yaxis=dict(range=[0, g2_y_max]),
        hovermode="x unified",
        template="simple_white",
        height=600,
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
    )
    st.plotly_chart(fig2, use_container_width=True)
