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

def add_reference_planes(fig, x_r, y_r, z_r, show_z=True):
    # X=0 Plane (Red)
    fig.add_trace(go.Surface(x=[0, 0], y=[y_r[0], y_r[1]], z=np.array([[z_r[0], z_r[1]], [z_r[0], z_r[1]]]), 
                             opacity=0.2, colorscale=[[0, 'red'], [1, 'red']], showscale=False, hoverinfo='skip'))
    # Y=0 Plane (Blue)
    fig.add_trace(go.Surface(x=[x_r[0], x_r[1]], y=[0, 0], z=np.array([[z_r[0], z_r[1]], [z_r[0], z_r[1]]]).T, 
                             opacity=0.2, colorscale=[[0, 'blue'], [1, 'blue']], showscale=False, hoverinfo='skip'))
    # Z=0 Plane (Yellow)
    if show_z:
        fig.add_trace(go.Surface(x=[x_r[0], x_r[1]], y=[y_r[0], y_r[1]], z=np.zeros((2,2)), 
                                 opacity=0.2, colorscale=[[0, 'yellow'], [1, 'yellow']], showscale=False, hoverinfo='skip'))

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Page 1: Definitions & Examples", "Page 2: 3D Analysis & Derivatives", "Page 3: Real-Time Level Curves"])

x_s, y_s = sp.symbols('x y')
presets = {"Linear": "x + y", "Rational": "5/(x**2 + y**2 + 1)", "Root": "sqrt(x**2 + y**2)", "Trigo": "sin(x)*cos(y)"}

# ---------------------------------------------------------
# PAGE 1: DEFINITIONS & EXAMPLES
# ---------------------------------------------------------
if page == "Page 1: Definitions & Examples":
    st.title("📖 Page 1: Functions of Two Variables")

    # --- NEW DEFINITION SECTION ---
    st.info("### 📘 Mathematical Definition")
    st.markdown("""
    A **function of two variables** is a rule that assigns to each ordered pair of real numbers 
    $(x, y)$ in a set $D$ (the domain) a unique real number denoted by $f(x, y)$. 
    The graph of such a function is a surface in 3D space.
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
            f_p = smart_parse(formula); f_n = sp.lambdify((x_s, y_s), f_p, 'numpy')
            px = np.linspace(-3, 3, 30); py = np.linspace(-3, 3, 30); PX, PY = np.meshgrid(px, py); PZ = f_n(PX, PY)
            fig = go.Figure(data=[go.Surface(z=PZ, x=PX, y=PY, showscale=False)])
            add_reference_planes(fig, [-3, 3], [-3, 3], [np.nanmin(PZ), np.nanmax(PZ)])
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# PAGE 2: 3D ANALYSIS (Expanded Tangents & Gradient)
# ---------------------------------------------------------
elif page == "Page 2: 3D Analysis & Derivatives":
    st.title("🧊 Page 2: Tangent Lines & Gradient Analysis")
    
    func_type = st.sidebar.selectbox("Function Mode:", ["Custom"] + list(presets.keys()))
    user_input = st.sidebar.text_input("Function f(x,y):", "x**2 - y**2") if func_type == "Custom" else presets[func_type]
    
    x_min, x_max = st.sidebar.slider("X Range", -10.0, 10.0, (-5.0, 5.0))
    y_min, y_max = st.sidebar.slider("Y Range", -10.0, 10.0, (-5.0, 5.0))
    
    mode = st.sidebar.radio("Mode:", ["Standard", "Analyse"])
    
    if mode == "Analyse":
        st.sidebar.subheader("Point Selection (x0, y0)")
        px0 = st.sidebar.number_input("x coordinate", value=0.0)
        py0 = st.sidebar.number_input("y coordinate", value=0.0)
        show_dx = st.sidebar.checkbox("Show ∂f/∂x (Red Line)", value=True)
        show_dy = st.sidebar.checkbox("Show ∂f/∂y (Blue Line)", value=True)
        show_grad = st.sidebar.checkbox("Show Gradient Tangent Plane (Purple)", value=True)

    try:
        f_s = smart_parse(user_input)
        df_dx = sp.diff(f_s, x_s); df_dy = sp.diff(f_s, y_s)
        f_np = sp.lambdify((x_s, y_s), f_s, 'numpy')

        x_v = np.linspace(x_min, x_max, 50); y_v = np.linspace(y_min, y_max, 50)
        X, Y = np.meshgrid(x_v, y_v); Z = f_np(X, Y)

        fig = go.Figure()
        add_reference_planes(fig, [x_min, x_max], [y_min, y_max], [np.nanmin(Z), np.nanmax(Z)])
        
        main_opacity = 1.0 if mode == "Standard" else 0.4
        fig.add_trace(go.Surface(z=Z, x=X, y=Y, opacity=main_opacity, colorscale='Viridis', name='f(x,y)'))

        if mode == "Analyse":
            z0 = float(f_s.subs({x_s: px0, y_s: py0}))
            slope_x = float(df_dx.subs({x_s: px0, y_s: py0}))
            slope_y = float(df_dy.subs({x_s: px0, y_s: py0}))

            if show_dx:
                tx = np.linspace(x_min, x_max, 50)
                tz_x = z0 + slope_x * (tx - px0)
                fig.add_trace(go.Scatter3d(x=tx, y=[py0]*50, z=tz_x, mode='lines', line=dict(color='red', width=12), name='∂f/∂x'))

            if show_dy:
                ty = np.linspace(y_min, y_max, 50)
                tz_y = z0 + slope_y * (ty - py0)
                fig.add_trace(go.Scatter3d(x=[px0]*50, y=ty, z=tz_y, mode='lines', line=dict(color='blue', width=12), name='∂f/∂y'))
            
            if show_grad:
                GZ = z0 + slope_x*(X - px0) + slope_y*(Y - py0)
                fig.add_trace(go.Surface(z=GZ, x=X, y=Y, opacity=0.9, colorscale=[[0, 'purple'], [1, 'purple']], showscale=False, name='Full Tangent Plane'))

            fig.add_trace(go.Scatter3d(x=[px0], y=[py0], z=[z0], mode='markers', marker=dict(size=10, color='black')))

        st.plotly_chart(fig, use_container_width=True)
        if mode == "Analyse":
            st.latex(f"f({px0}, {py0}) = {z0:.2f} \\quad \\nabla f({px0}, {py0}) = \\langle {slope_x:.2f}, {slope_y:.2f} \\rangle")

    except Exception as e:
        st.error(f"Error: {e}")

# ---------------------------------------------------------
# PAGE 3: LEVEL CURVES
# ---------------------------------------------------------
elif page == "Page 3: Real-Time Level Curves":
    st.title("🗺️ Page 3: 3D Intersection Level Curves")
    import matplotlib.pyplot as plt 

    func_type = st.sidebar.selectbox("Function Mode:", ["Custom"] + list(presets.keys()))
    user_input = st.sidebar.text_input("Function f(x,y):", "x**2 + y**2") if func_type == "Custom" else presets[func_type]
    
    x_min, x_max = st.sidebar.slider("X Range", -10.0, 10.0, (-5.0, 5.0))
    y_min, y_max = st.sidebar.slider("Y Range", -10.0, 10.0, (-5.0, 5.0))
    
    f_s = smart_parse(user_input); f_np = sp.lambdify((x_s, y_s), f_s, 'numpy')
    x_v = np.linspace(x_min, x_max, 80); y_v = np.linspace(y_min, y_max, 80)
    X, Y = np.meshgrid(x_v, y_v); Z = f_np(X, Y)
    
    k_val = st.sidebar.slider("Adjust Level k (z = k)", float(np.nanmin(Z)), float(np.nanmax(Z)), float(np.nanmean(Z)))

    fig = go.Figure()
    add_reference_planes(fig, [x_min, x_max], [y_min, y_max], [np.nanmin(Z), np.nanmax(Z)], show_z=False)
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, opacity=0.5, colorscale='Viridis', showscale=False))
    
    K_plane = np.full_like(Z, k_val)
    fig.add_trace(go.Surface(z=K_plane, x=X, y=Y, opacity=0.9, colorscale=[[0, 'yellow'], [1, 'yellow']], showscale=False))

    try:
        cs = plt.contour(X, Y, Z, levels=[k_val])
        for collection in cs.collections:
            for path in collection.get_paths():
                v = path.vertices
                fig.add_trace(go.Scatter3d(x=v[:, 0], y=v[:, 1], z=np.full(len(v), k_val), 
                                           mode='lines', line=dict(color='black', width=12), name='Level Curve'))
        plt.close()
    except:
        st.warning("Intersection path not found for this k-level.")

    fig.update_layout(scene=dict(aspectmode='cube'))
    st.plotly_chart(fig, use_container_width=True)
