import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
# Make sure to run: pip install streamlit-plotly-events
from streamlit_plotly_events import plotly_events 

# --- APP CONFIGURATION ---
st.set_page_config(page_title="MAT201 Calculus Explorer", layout="wide")

# Initialize Session State
if 'px0' not in st.session_state:
    st.session_state.px0 = 0.0
if 'py0' not in st.session_state:
    st.session_state.py0 = 0.0

def smart_parse(user_input):
    try:
        clean_input = user_input.replace('^', '**')
        transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))
        return parse_expr(clean_input, transformations=transformations)
    except:
        return None

def add_reference_planes(fig, x_r, y_r, z_r, show_z=True):
    # Planes are added with very low opacity to stay in background
    fig.add_trace(go.Surface(x=[0, 0], y=[y_r[0], y_r[1]], z=np.array([[z_r[0], z_r[1]], [z_r[0], z_r[1]]]), 
                             opacity=0.1, colorscale=[[0, 'red'], [1, 'red']], showscale=False, hoverinfo='skip'))
    fig.add_trace(go.Surface(x=[x_r[0], x_r[1]], y=[0, 0], z=np.array([[z_r[0], z_r[1]], [z_r[0], z_r[1]]]).T, 
                             opacity=0.1, colorscale=[[0, 'blue'], [1, 'blue']], showscale=False, hoverinfo='skip'))
    if show_z:
        fig.add_trace(go.Surface(x=[x_r[0], x_r[1]], y=[y_r[0], y_r[1]], z=np.zeros((2,2)), 
                                 opacity=0.1, colorscale=[[0, 'yellow'], [1, 'yellow']], showscale=False, hoverinfo='skip'))

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Page 1: Definitions & Examples", "Page 2: 3D Analysis & Derivatives"])

x_s, y_s = sp.symbols('x y')
presets = {"Linear": "x + y", "Rational": "5/(x**2 + y**2 + 1)", "Root": "sqrt(x**2 + y**2)", "Trigo": "sin(x)*cos(y)"}

# ---------------------------------------------------------
# PAGE 1: DEFINITIONS & EXAMPLES
# ---------------------------------------------------------
if page == "Page 1: Definitions & Examples":
    st.title("📖 Page 1: Functions of Two Variables")
    st.info("### 📘 Mathematical Definition")
    st.markdown("A function of two variables assigns a unique $z$ to every $(x, y)$.")
    
    cols = st.columns(2)
    for idx, (name, formula) in enumerate(presets.items()):
        with cols[idx % 2]:
            st.write(f"**{name}:** $f(x,y) = {formula}$")
            f_p = smart_parse(formula)
            if f_p:
                f_n = sp.lambdify((x_s, y_s), f_p, 'numpy')
                px = np.linspace(-3, 3, 30); py = np.linspace(-3, 3, 30)
                PX, PY = np.meshgrid(px, py); PZ = f_n(PX, PY)
                fig_eg = go.Figure(data=[go.Surface(z=PZ, x=PX, y=PY, showscale=False)])
                add_reference_planes(fig_eg, [-3, 3], [-3, 3], [np.nanmin(PZ), np.nanmax(PZ)])
                st.plotly_chart(fig_eg, use_container_width=True)

# ---------------------------------------------------------
# PAGE 2: 3D ANALYSIS & DERIVATIVES (THE FIX)
# ---------------------------------------------------------
elif page == "Page 2: 3D Analysis & Derivatives":
    st.title("🧊 Page 2: Tangent Lines & Gradient Analysis")
    
    # Inputs
    func_type = st.sidebar.selectbox("Function Mode:", ["Custom"] + list(presets.keys()))
    user_input = st.sidebar.text_input("Function f(x,y):", "x**2 - y**2") if func_type == "Custom" else presets[func_type]
    x_min, x_max = st.sidebar.slider("X Range", -10.0, 10.0, (-5.0, 5.0))
    y_min, y_max = st.sidebar.slider("Y Range", -10.0, 10.0, (-5.0, 5.0))
    mode = st.sidebar.radio("Mode:", ["Standard", "Analyse"])

    try:
        f_s = smart_parse(user_input)
        if f_s:
            # 1. Prepare Data
            df_dx = sp.diff(f_s, x_s); df_dy = sp.diff(f_s, y_s)
            f_np = sp.lambdify((x_s, y_s), f_s, 'numpy')
            x_v = np.linspace(x_min, x_max, 50); y_v = np.linspace(y_min, y_max, 50)
            X, Y = np.meshgrid(x_v, y_v); Z = f_np(X, Y)

            # 2. Build Figure from scratch every rerun
            fig = go.Figure()
            
            # --- FORCE MAIN SURFACE TO BE SOLID ---
            fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis', opacity=1.0, name='Surface', showscale=True))

            if mode == "Analyse":
                st.sidebar.subheader("Point Selection")
                st.session_state.px0 = st.sidebar.number_input("x", value=float(st.session_state.px0), step=0.1)
                st.session_state.py0 = st.sidebar.number_input("y", value=float(st.session_state.py0), step=0.1)
                
                curr_x, curr_y = st.session_state.px0, st.session_state.py0
                z0 = float(f_s.subs({x_s: curr_x, y_s: curr_y}))
                sx = float(df_dx.subs({x_s: curr_x, y_s: curr_y}))
                sy = float(df_dy.subs({x_s: curr_x, y_s: curr_y}))

                # Red Tangent (X)
                tx = np.linspace(x_min, x_max, 50)
                fig.add_trace(go.Scatter3d(x=tx, y=[curr_y]*50, z=z0 + sx*(tx-curr_x), mode='lines', line=dict(color='red', width=10)))
                # Blue Tangent (Y)
                ty = np.linspace(y_min, y_max, 50)
                fig.add_trace(go.Scatter3d(x=[curr_x]*50, y=ty, z=z0 + sy*(ty-curr_y), mode='lines', line=dict(color='blue', width=10)))
                # Purple Plane
                GZ = z0 + sx*(X - curr_x) + sy*(Y - curr_y)
                fig.add_trace(go.Surface(z=GZ, x=X, y=Y, opacity=0.5, colorscale=[[0, 'purple'], [1, 'purple']], showscale=False))
                # Black Point
                fig.add_trace(go.Scatter3d(x=[curr_x], y=[curr_y], z=[z0], mode='markers', marker=dict(size=10, color='black')))

                # 3. INTERACTIVE RENDER
                # Unique key prevents the "vanishing" bug by forcing a clean redraw
                plot_key = f"plot_{user_input}_{mode}_{x_min}_{y_min}"
                selected = plotly_events(fig, click_event=True, hover_event=False, key=plot_key)
                
                if selected:
                    st.session_state.px0 = float(selected[0]['x'])
                    st.session_state.py0 = float(selected[0]['y'])
                    st.rerun()

                st.latex(f"f({curr_x:.2f}, {curr_y:.2f}) = {z0:.2f} \\quad \\nabla f = \\langle {sx:.2f}, {sy:.2f} \\rangle")
            
            else:
                add_reference_planes(fig, [x_min, x_max], [y_min, y_max], [np.nanmin(Z), np.nanmax(Z)])
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
