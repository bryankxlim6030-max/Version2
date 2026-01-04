import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
# You must install this: pip install streamlit-math-live streamlit-plotly-events
from streamlit_math_live import math_live
from streamlit_plotly_events import plotly_events 

# --- APP CONFIGURATION ---
st.set_page_config(page_title="MAT201 Calculus Explorer", layout="wide")

# Initialize Session State for coordinates
if 'px0' not in st.session_state:
    st.session_state.px0 = 0.0
if 'py0' not in st.session_state:
    st.session_state.py0 = 0.0

def smart_parse(user_input):
    try:
        # MathLive provides LaTeX, so we convert common LaTeX to SymPy readable format
        clean_input = user_input.replace(r'\frac', '/').replace('{', '(').replace('}', ')')
        clean_input = clean_input.replace(r'\cdot', '*').replace('^', '**')
        # Removing remaining backslashes from common functions
        for func in ['sin', 'cos', 'tan', 'sqrt', 'log', 'exp']:
            clean_input = clean_input.replace('\\' + func, func)
        
        transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))
        return parse_expr(clean_input, transformations=transformations)
    except:
        return None

def add_reference_planes(fig, x_r, y_r, z_r, show_z=True):
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
# PAGE 2: 3D ANALYSIS (STABLE VERSION - ANALYSE ONLY)
# ---------------------------------------------------------
elif page == "Page 2: 3D Analysis & Derivatives":
    st.title("🧊 Page 2: Tangent Lines & Gradient Analysis")
    st.caption("Use the math keyboard below to define your function, then click the surface to analyze points.")

    # Function Input with MathLive
    st.write("### Define Function $f(x,y)$")
    func_type = st.sidebar.selectbox("Preset Functions:", ["Custom"] + list(presets.keys()))
    
    # MathLive Keyboard Integration
    default_val = "x^2 - y^2" if func_type == "Custom" else presets[func_type].replace("**", "^")
    user_input = math_live(value=default_val, key="math_input")
    
    x_min, x_max = st.sidebar.slider("X Range", -10.0, 10.0, (-5.0, 5.0))
    y_min, y_max = st.sidebar.slider("Y Range", -10.0, 10.0, (-5.0, 5.0))

    # Point Selection
    st.sidebar.subheader("Point Selection")
    st.session_state.px0 = st.sidebar.number_input("x coordinate", value=float(st.session_state.px0), step=0.1)
    st.session_state.py0 = st.sidebar.number_input("y coordinate", value=float(st.session_state.py0), step=0.1)
    
    show_dx = st.sidebar.checkbox("Show ∂f/∂x (Red)", value=True)
    show_dy = st.sidebar.checkbox("Show ∂f/∂y (Blue)", value=True)
    show_grad = st.sidebar.checkbox("Show Tangent Plane (Purple)", value=True)

    try:
        f_s = smart_parse(user_input)
        if f_s:
            df_dx = sp.diff(f_s, x_s); df_dy = sp.diff(f_s, y_s)
            f_np = sp.lambdify((x_s, y_s), f_s, 'numpy')

            x_v = np.linspace(x_min, x_max, 40); y_v = np.linspace(y_min, y_max, 40)
            X, Y = np.meshgrid(x_v, y_v); Z = f_np(X, Y)

            fig = go.Figure()
            
            # Surface rendering
            fig.add_trace(go.Mesh3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), 
                                    opacity=0.6, color='lightblue', name='Surface'))

            curr_x, curr_y = st.session_state.px0, st.session_state.py0
            z0 = float(f_s.subs({x_s: curr_x, y_s: curr_y}))
            sx = float(df_dx.subs({x_s: curr_x, y_s: curr_y}))
            sy = float(df_dy.subs({x_s: curr_x, y_s: curr_y}))

            if show_dx:
                tx = np.linspace(x_min, x_max, 40)
                fig.add_trace(go.Scatter3d(x=tx, y=[curr_y]*40, z=z0 + sx*(tx-curr_x), mode='lines', line=dict(color='red', width=10)))
            if show_dy:
                ty = np.linspace(y_min, y_max, 40)
                fig.add_trace(go.Scatter3d(x=[curr_x]*40, y=ty, z=z0 + sy*(ty-curr_y), mode='lines', line=dict(color='blue', width=10)))
            if show_grad:
                GZ = z0 + sx*(X - curr_x) + sy*(Y - curr_y)
                fig.add_trace(go.Surface(z=GZ, x=X, y=Y, opacity=0.4, colorscale=[[0, 'purple'], [1, 'purple']], showscale=False))
            
            fig.add_trace(go.Scatter3d(x=[curr_x], y=[curr_y], z=[z0], mode='markers', marker=dict(size=10, color='black')))

            # Click Event Integration
            selected = plotly_events(fig, click_event=True, key=f"plot_{user_input}")
            
            if selected:
                st.session_state.px0 = float(selected[0]['x'])
                st.session_state.py0 = float(selected[0]['y'])
                st.rerun()

            st.latex(f"f({curr_x:.2f}, {curr_y:.2f}) = {z0:.2f}")
            st.latex(r"\nabla f = \langle" + f"{sx:.2f}, {sy:.2f}" + r"\rangle")

    except Exception as e:
        st.error(f"Mathematical Error: {e}")
