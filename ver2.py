import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
# Make sure to run: pip install streamlit-plotly-events
from streamlit_plotly_events import plotly_events 

# --- APP CONFIGURATION ---
st.set_page_config(page_title="MAT201 Calculus Explorer", layout="wide")

# Initialize Session State for coordinates and function tracking
if 'px0' not in st.session_state:
    st.session_state.px0 = 0.0
if 'py0' not in st.session_state:
    st.session_state.py0 = 0.0
if 'last_func' not in st.session_state:
    st.session_state.last_func = ""

def smart_parse(user_input):
    try:
        clean_input = user_input.replace('^', '**')
        transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))
        return parse_expr(clean_input, transformations=transformations)
    except:
        return None

def add_reference_planes(fig, x_r, y_r, z_r, show_z=True):
    # X=0 Plane (Red)
    fig.add_trace(go.Surface(x=[0, 0], y=[y_r[0], y_r[1]], z=np.array([[z_r[0], z_r[1]], [z_r[0], z_r[1]]]), 
                             opacity=0.1, colorscale=[[0, 'red'], [1, 'red']], showscale=False, hoverinfo='skip'))
    # Y=0 Plane (Blue)
    fig.add_trace(go.Surface(x=[x_r[0], x_r[1]], y=[0, 0], z=np.array([[z_r[0], z_r[1]], [z_r[0], z_r[1]]]).T, 
                             opacity=0.1, colorscale=[[0, 'blue'], [1, 'blue']], showscale=False, hoverinfo='skip'))
    # Z=0 Plane (Yellow)
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
    st.markdown("""
    A **function of two variables** is a rule that assigns to each ordered pair of real numbers 
    $(x, y)$ in a set $D$ (the domain) a unique real number denoted by $f(x, y)$. 
    """)
    
    col_def1, col_def2 = st.columns(2)
    with col_def1:
        st.write("**General Form:**")
        st.latex(r"z = f(x, y)")
    with col_def2:
        st.write("**Geometric Interpretation:**")
        st.write("The set of all points $(x, y, z)$ such that $z = f(x, y)$ forms a **Surface** in $\mathbb{R}^3$.")

    st.markdown("---")
    st.subheader("💡 Visualizing Common Surface Examples")

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
# PAGE 2: 3D ANALYSIS & DERIVATIVES
# ---------------------------------------------------------
elif page == "Page 2: 3D Analysis & Derivatives":
    st.title("🧊 Page 2: Tangent Lines & Gradient Analysis")
    st.caption("Click the surface to select a point, or use the sidebar for manual entry.")

    func_type = st.sidebar.selectbox("Function Mode:", ["Custom"] + list(presets.keys()))
    user_input = st.sidebar.text_input("Function f(x,y):", "x**2 - y**2") if func_type == "Custom" else presets[func_type]
    
    x_min, x_max = st.sidebar.slider("X Range", -10.0, 10.0, (-5.0, 5.0))
    y_min, y_max = st.sidebar.slider("Y Range", -10.0, 10.0, (-5.0, 5.0))
    mode = st.sidebar.radio("Mode:", ["Standard", "Analyse"])

    if st.session_state.last_func != user_input:
        st.session_state.last_func = user_input

    if mode == "Analyse":
        st.sidebar.subheader("Point Selection")
        st.session_state.px0 = st.sidebar.number_input("x coordinate", value=float(st.session_state.px0), step=0.1)
        st.session_state.py0 = st.sidebar.number_input("y coordinate", value=float(st.session_state.py0), step=0.1)
        
        show_dx = st.sidebar.checkbox("Show ∂f/∂x (Red)", value=True)
        show_dy = st.sidebar.checkbox("Show ∂f/∂y (Blue)", value=True)
        show_grad = st.sidebar.checkbox("Show Tangent Plane (Purple)", value=True)

    try:
        f_s = smart_parse(user_input)
        if f_s is not None:
            df_dx = sp.diff(f_s, x_s)
            df_dy = sp.diff(f_s, y_s)
            f_np = sp.lambdify((x_s, y_s), f_s, 'numpy')

            x_v = np.linspace(x_min, x_max, 50)
            y_v = np.linspace(y_min, y_max, 50)
            X, Y = np.meshgrid(x_v, y_v)
            Z = f_np(X, Y)

            fig = go.Figure()
            
            # 1. Always add the main surface first
            # Increased opacity to 0.8 in Analyse mode to prevent it from disappearing
            main_op = 1.0 if mode == "Standard" else 0.8
            fig.add_trace(go.Surface(z=Z, x=X, y=Y, opacity=main_op, colorscale='Viridis', name='Main Surface'))

            if mode == "Analyse":
                curr_x, curr_y = st.session_state.px0, st.session_state.py0
                z0 = float(f_s.subs({x_s: curr_x, y_s: curr_y}))
                slope_x = float(df_dx.subs({x_s: curr_x, y_s: curr_y}))
                slope_y = float(df_dy.subs({x_s: curr_x, y_s: curr_y}))

                # Add analysis traces
                if show_dx:
                    tx = np.linspace(x_min, x_max, 50)
                    fig.add_trace(go.Scatter3d(x=tx, y=[curr_y]*50, z=z0 + slope_x*(tx-curr_x), mode='lines', line=dict(color='red', width=10)))
                if show_dy:
                    ty = np.linspace(y_min, y_max, 50)
                    fig.add_trace(go.Scatter3d(x=[curr_x]*50, y=ty, z=z0 + slope_y*(ty-curr_y), mode='lines', line=dict(color='blue', width=10)))
                if show_grad:
                    GZ = z0 + slope_x*(X - curr_x) + slope_y*(Y - curr_y)
                    fig.add_trace(go.Surface(z=GZ, x=X, y=Y, opacity=0.4, colorscale=[[0, 'purple'], [1, 'purple']], showscale=False))
                
                fig.add_trace(go.Scatter3d(x=[curr_x], y=[curr_y], z=[z0], mode='markers', marker=dict(size=8, color='black')))

                # Add reference planes last so they are behind
                add_reference_planes(fig, [x_min, x_max], [y_min, y_max], [np.nanmin(Z), np.nanmax(Z)])

                # Render with events
                selected = plotly_events(fig, click_event=True, hover_event=False, key=f"int_{user_input}")
                if selected:
                    st.session_state.px0 = float(selected[0]['x'])
                    st.session_state.py0 = float(selected[0]['y'])
                    st.rerun()

                st.markdown("---")
                st.latex(f"f({curr_x:.2f}, {curr_y:.2f}) = {z0:.2f}")
                st.latex(r"\nabla f = \langle " + f"{slope_x:.2f}, {slope_y:.2f}" + r"\rangle")
            
            else:
                add_reference_planes(fig, [x_min, x_max], [y_min, y_max], [np.nanmin(Z), np.nanmax(Z)])
                st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error: {e}")
