import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

# --- APP CONFIGURATION ---
st.set_page_config(page_title="MAT201 Calculus Explorer", layout="wide")

def smart_parse(user_input):
    try:
        clean_input = user_input.replace('^', '**')
        transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))
        return parse_expr(clean_input, transformations=transformations)
    except:
        return None

def add_reference_planes(fig, x_range, y_range, z_range, show_z=True):
    # X=0 Plane (Red)
    fig.add_trace(go.Surface(x=[0, 0], y=y_range, z=np.array([z_range, z_range]).T, 
                             opacity=0.1, colorscale=[[0, 'red'], [1, 'red']], showscale=False, hoverinfo='skip'))
    # Y=0 Plane (Blue)
    fig.add_trace(go.Surface(x=x_range, y=[0, 0], z=np.array([z_range, z_range]), 
                             opacity=0.1, colorscale=[[0, 'blue'], [1, 'blue']], showscale=False, hoverinfo='skip'))
    # Z=0 Plane (Yellow)
    if show_z:
        fig.add_trace(go.Surface(x=x_range, y=y_range, z=np.zeros((2,2)), 
                                 opacity=0.1, colorscale=[[0, 'yellow'], [1, 'yellow']], showscale=False, hoverinfo='skip'))

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Page 1: Definitions & Examples", "Page 2: 3D Analysis & Derivatives", "Page 3: Real-Time Level Curves"])

x_s, y_s = sp.symbols('x y')
presets = {"Linear": "x + y", "Rational": "5/(x**2 + y**2 + 1)", "Root": "sqrt(x**2 + y**2)", "Trigo": "sin(x)*cos(y)"}

# ---------------------------------------------------------
# PAGE 1: DEFINITIONS & EXAMPLES (Reverted to Full View)
# ---------------------------------------------------------
if page == "Page 1: Definitions & Examples":
    st.title("📖 Page 1: Understanding Functions of Two Variables")
    st.markdown("### Definition: $z = f(x, y)$\nA function of two variables assigns a unique real number to each pair $(x, y)$.")
    
    cols = st.columns(2)
    for idx, (name, formula) in enumerate(presets.items()):
        with cols[idx % 2]:
            st.write(f"**{name}:** $f(x,y) = {formula}$")
            f_p = smart_parse(formula)
            f_n = sp.lambdify((x_s, y_s), f_p, 'numpy')
            res = 30
            px = np.linspace(-3, 3, res); py = np.linspace(-3, 3, res)
            PX, PY = np.meshgrid(px, py); PZ = f_n(PX, PY)
            fig = go.Figure(data=[go.Surface(z=PZ, x=PX, y=PY, showscale=False)])
            add_reference_planes(fig, [-3, 3], [-3, 3], [np.nanmin(PZ), np.nanmax(PZ)])
            fig.update_layout(height=400, margin=dict(l=0,r=0,b=0,t=0))
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# PAGE 2: 3D ANALYSIS & MODES
# ---------------------------------------------------------
elif page == "Page 2: 3D Analysis & Derivatives":
    st.title("🧊 Page 2: Advanced 3D Analysis")
    
    func_type = st.sidebar.selectbox("Function Mode:", ["Custom", "Linear", "Rational", "Root", "Trigo"])
    user_input = st.sidebar.text_input("Function f(x,y):", presets["Linear"] if func_type == "Linear" else "x**2 - y**2") if func_type == "Custom" else presets[func_type]
    
    x_min, x_max = st.sidebar.slider("X Range", -10.0, 10.0, (-5.0, 5.0))
    y_min, y_max = st.sidebar.slider("Y Range", -10.0, 10.0, (-5.0, 5.0))
    
    mode = st.sidebar.radio("View Mode:", ["Standard", "Partial Derivatives", "Gradient"])
    show_details = st.sidebar.checkbox("Show Detailed Analysis (Click Simulator)", value=False)

    try:
        f_s = smart_parse(user_input)
        df_dx = sp.diff(f_s, x_s); df_dy = sp.diff(f_s, y_s)
        f_np = sp.lambdify((x_s, y_s), f_s, 'numpy')

        x_v = np.linspace(x_min, x_max, 40); y_v = np.linspace(y_min, y_max, 40)
        X, Y = np.meshgrid(x_v, y_v); Z = f_np(X, Y)

        fig = go.Figure()
        add_reference_planes(fig, [x_min, x_max], [y_min, y_max], [np.nanmin(Z), np.nanmax(Z)])

        opac = 1.0 if mode == "Standard" else 0.3
        fig.add_trace(go.Surface(z=Z, x=X, y=Y, opacity=opac, colorscale='Viridis'))

        if show_details:
            if mode == "Partial Derivatives":
                dz_dx_np = sp.lambdify((x_s, y_s), df_dx, 'numpy')
                Z_d = dz_dx_np(X, Y)
                if np.isscalar(Z_d): Z_d = np.full_like(X, Z_d)
                fig.add_trace(go.Surface(z=Z_d, x=X, y=Y, opacity=0.9, colorscale='Reds'))
            elif mode == "Gradient":
                grad_mag = sp.sqrt(df_dx**2 + df_dy**2)
                g_np = sp.lambdify((x_s, y_s), grad_mag, 'numpy')
                Z_g = g_np(X, Y)
                if np.isscalar(Z_g): Z_g = np.full_like(X, Z_g)
                fig.add_trace(go.Surface(z=Z_g, x=X, y=Y, opacity=0.9, colorscale='Magma'))

        st.plotly_chart(fig, use_container_width=True)
        st.latex(f"f(x,y) = {sp.latex(f_s)}")
        st.latex(f"\\frac{{\\partial f}}{{\\partial x}} = {sp.latex(df_dx)} \\quad \\frac{{\\partial f}}{{\\partial y}} = {sp.latex(df_dy)}")

    except Exception as e:
        st.error(f"Error: {e}")

# ---------------------------------------------------------
# PAGE 3: LEVEL CURVES
# ---------------------------------------------------------
elif page == "Page 3: Real-Time Level Curves":
    st.title("🗺️ Page 3: Real-Time Level Surface Intersection")
    
    func_type = st.sidebar.selectbox("Function Mode:", ["Custom", "Linear", "Rational", "Root", "Trigo"])
    user_input = st.sidebar.text_input("Function f(x,y):", "x**2 + y**2") if func_type == "Custom" else presets[func_type]
    
    x_min, x_max = st.sidebar.slider("X Range", -10.0, 10.0, (-5.0, 5.0))
    y_min, y_max = st.sidebar.slider("Y Range", -10.0, 10.0, (-5.0, 5.0))
    
    f_s = smart_parse(user_input)
    f_np = sp.lambdify((x_s, y_s), f_s, 'numpy')
    x_v = np.linspace(x_min, x_max, 50); y_v = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(x_v, y_v); Z = f_np(X, Y)
    
    z_min, z_max = np.nanmin(Z), np.nanmax(Z)
    k_val = st.sidebar.slider("Adjust Yellow Surface k (z = k)", float(z_min), float(z_max), float((z_min+z_max)/2))

    fig = go.Figure()
    # Reference X=0 (Red) and Y=0 (Blue) only
    add_reference_planes(fig, [x_min, x_max], [y_min, y_max], [z_min, z_max], show_z=False)
    
    # f(x,y) at 50% transparency
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, opacity=0.5, colorscale='Viridis', showscale=False))
    
    # Yellow Shifting Surface z=k
    K_plane = np.full_like(Z, k_val)
    fig.add_trace(go.Surface(z=K_plane, x=X, y=Y, opacity=0.5, colorscale=[[0, 'yellow'], [1, 'yellow']], showscale=False))

    # Bolded Black Intersection Line (Level Curve)
    fig.add_trace(go.Contour(z=Z, x=x_v, y=y_v, contours=dict(start=k_val, end=k_val, size=0.1),
                             line=dict(width=10, color='black'), showscale=False))

    fig.update_layout(scene=dict(aspectmode='cube'))
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"The bold black line shows the level curve where $f(x,y) = {k_val}$.")
