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
    # X=0 Plane (Red) - 20% Opacity
    fig.add_trace(go.Surface(x=[0, 0], y=[y_r[0], y_r[1]], z=np.array([[z_r[0], z_r[1]], [z_r[0], z_r[1]]]), 
                             opacity=0.2, colorscale=[[0, 'red'], [1, 'red']], showscale=False, hoverinfo='skip'))
    # Y=0 Plane (Blue) - 20% Opacity
    fig.add_trace(go.Surface(x=[x_r[0], x_r[1]], y=[0, 0], z=np.array([[z_r[0], z_r[1]], [z_r[0], z_r[1]]]).T, 
                             opacity=0.2, colorscale=[[0, 'blue'], [1, 'blue']], showscale=False, hoverinfo='skip'))
    # Z=0 Plane (Yellow) - 20% Opacity
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
    st.title("📖 Page 1: Definitions & Examples")
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
# PAGE 2: 3D ANALYSIS (Tangent Lines & Point Analysis)
# ---------------------------------------------------------
elif page == "Page 2: 3D Analysis & Derivatives":
    st.title("🧊 Page 2: Tangent Lines & Gradient Analysis")
    
    func_type = st.sidebar.selectbox("Function Mode:", ["Custom"] + list(presets.keys()))
    user_input = st.sidebar.text_input("Function f(x,y):", "x**2 - y**2") if func_type == "Custom" else presets[func_type]
    
    x_min, x_max = st.sidebar.slider("X Range", -10.0, 10.0, (-5.0, 5.0))
    y_min, y_max = st.sidebar.slider("Y Range", -10.0, 10.0, (-5.0, 5.0))
    
    st.sidebar.subheader("Point Analysis (x0, y0)")
    px0 = st.sidebar.number_input("x coordinate", value=0.0)
    py0 = st.sidebar.number_input("y coordinate", value=0.0)
    show_tangents = st.sidebar.checkbox("Show Tangent Lines & Gradient", value=True)

    try:
        f_s = smart_parse(user_input)
        df_dx = sp.diff(f_s, x_s); df_dy = sp.diff(f_s, y_s)
        f_np = sp.lambdify((x_s, y_s), f_s, 'numpy')

        x_v = np.linspace(x_min, x_max, 50); y_v = np.linspace(y_min, y_max, 50)
        X, Y = np.meshgrid(x_v, y_v); Z = f_np(X, Y)

        fig = go.Figure()
        add_reference_planes(fig, [x_min, x_max], [y_min, y_max], [np.nanmin(Z), np.nanmax(Z)])
        fig.add_trace(go.Surface(z=Z, x=X, y=Y, opacity=0.7, colorscale='Viridis', name='f(x,y)'))

        if show_tangents:
            # Point value and slopes
            z0 = float(f_s.subs({x_s: px0, y_s: py0}))
            slope_x = float(df_dx.subs({x_s: px0, y_s: py0}))
            slope_y = float(df_dy.subs({x_s: px0, y_s: py0}))

            # Tangent Line in X-direction
            tx = np.linspace(px0-2, px0+2, 10)
            tz_x = z0 + slope_x * (tx - px0)
            fig.add_trace(go.Scatter3d(x=tx, y=[py0]*10, z=tz_x, mode='lines', line=dict(color='red', width=8), name='Tangent w.r.t x'))

            # Tangent Line in Y-direction
            ty = np.linspace(py0-2, py0+2, 10)
            tz_y = z0 + slope_y * (ty - py0)
            fig.add_trace(go.Scatter3d(x=[px0]*10, y=ty, z=tz_y, mode='lines', line=dict(color='blue', width=8), name='Tangent w.r.t y'))
            
            # Point marker
            fig.add_trace(go.Scatter3d(x=[px0], y=[py0], z=[z0], mode='markers', marker=dict(size=8, color='black')))

        st.plotly_chart(fig, use_container_width=True)
        st.write(f"**Point Analysis at ({px0}, {py0}):**")
        st.latex(f"f({px0}, {py0}) = {z0:.2f}")
        st.latex(f"\\frac{{\\partial f}}{{\\partial x}} = {slope_x:.2f} \\quad \\frac{{\\partial f}}{{\\partial y}} = {slope_y:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")

# ---------------------------------------------------------
# PAGE 3: LEVEL CURVES (Fixed 3D Intersection)
# ---------------------------------------------------------
elif page == "Page 3: Real-Time Level Curves":
    st.title("🗺️ Page 3: 3D Intersection Level Curves")
    
    func_type = st.sidebar.selectbox("Function Selection:", ["Custom"] + list(presets.keys()))
    user_input = st.sidebar.text_input("Function f(x,y):", "x**2 + y**2") if func_type == "Custom" else presets[func_type]
    
    x_min, x_max = st.sidebar.slider("X Range", -10.0, 10.0, (-5.0, 5.0))
    y_min, y_max = st.sidebar.slider("Y Range", -10.0, 10.0, (-5.0, 5.0))
    
    f_s = smart_parse(user_input); f_np = sp.lambdify((x_s, y_s), f_s, 'numpy')
    x_v = np.linspace(x_min, x_max, 60); y_v = np.linspace(y_min, y_max, 60)
    X, Y = np.meshgrid(x_v, y_v); Z = f_np(X, Y)
    
    k_val = st.sidebar.slider("Adjust Shifting Plane (z = k)", float(np.nanmin(Z)), float(np.nanmax(Z)), float(np.nanmean(Z)))

    fig = go.Figure()
    add_reference_planes(fig, [x_min, x_max], [y_min, y_max], [np.nanmin(Z), np.nanmax(Z)], show_z=False)
    
    # f(x,y) surface
    fig.add_trace(go.Surface(z=Z, x=X, y=Y, opacity=0.5, colorscale='Viridis', showscale=False))
    
    # Yellow Shifting Plane
    K_plane = np.full_like(Z, k_val)
    fig.add_trace(go.Surface(z=K_plane, x=X, y=Y, opacity=0.3, colorscale=[[0, 'yellow'], [1, 'yellow']], showscale=False))

    # Calculate 3D Intersection Points
    # Find contours at the specific k-level
    import matplotlib.pyplot as plt
    cs = plt.contour(X, Y, Z, levels=[k_val])
    for path in cs.collections[0].get_paths():
        v = path.vertices
        fig.add_trace(go.Scatter3d(x=v[:, 0], y=v[:, 1], z=[k_val]*len(v), 
                                   mode='lines', line=dict(color='black', width=10), name='Intersection'))
    plt.close()

    fig.update_layout(scene=dict(aspectmode='cube'))
    st.plotly_chart(fig, use_container_width=True)
