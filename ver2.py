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

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Page 1: Definitions & Examples", "Page 2: 3D Analysis & Derivatives", "Page 3: Real-Time Level Curves"])

x_s, y_s = sp.symbols('x y')

# ---------------------------------------------------------
# PAGE 1: DEFINITIONS (First Octant Focus)
# ---------------------------------------------------------
if page == "Page 1: Definitions & Examples":
    st.title("📖 Page 1: Definitions (First Octant View)")
    st.markdown("### Definition: $z = f(x, y)$\nIn this view, we focus on the **First Octant** ($x, y, z > 0$).")
    
    cols = st.columns(2)
    examples = [("Linear", "x + y"), ("Rational", "5/(x*y + 1)"), ("Root", "sqrt(x*y)"), ("Trigo", "abs(sin(x)*cos(y)*5)")]
    
    for idx, (name, formula) in enumerate(examples):
        with cols[idx % 2]:
            st.write(f"**{name}:** $f(x,y) = {formula}$")
            f_p = smart_parse(formula)
            f_n = sp.lambdify((x_s, y_s), f_p, 'numpy')
            # First Octant Range: 0 to 5
            px = np.linspace(0.01, 5, 30); py = np.linspace(0.01, 5, 30)
            PX, PY = np.meshgrid(px, py)
            PZ = f_n(PX, PY)
            fig = go.Figure(data=[go.Surface(z=PZ, x=PX, y=PY, showscale=False)])
            fig.update_layout(height=300, margin=dict(l=0,r=0,b=0,t=0),
                              scene=dict(xaxis=dict(range=[0,5]), yaxis=dict(range=[0,5]), zaxis=dict(range=[0, max(5, np.max(PZ)) if not np.isinf(np.max(PZ)) else 5])))
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# PAGE 2: 3D SURFACE & MODES (Gradients & Derivatives)
# ---------------------------------------------------------
elif page == "Page 2: 3D Surface & Derivatives":
    st.title("🧊 Page 2: Advanced 3D Analysis")
    
    # Sidebar Controls
    st.sidebar.subheader("Function & Range")
    user_input = st.sidebar.text_input("Function f(x,y):", "x**2 - y**2")
    x_range = st.sidebar.slider("X/Y Limit", 1, 10, 5)
    
    st.sidebar.subheader("Visual Focus Mode")
    mode = st.sidebar.radio("Focus Overlay:", ["Standard Surface", "Partial Derivative Mode", "Gradient Surface Mode"])

    try:
        f_s = smart_parse(user_input)
        df_dx = sp.diff(f_s, x_s)
        df_dy = sp.diff(f_s, y_s)
        
        f_np = sp.lambdify((x_s, y_s), f_s, 'numpy')
        
        res = 50
        x_v = np.linspace(-x_range, x_range, res)
        y_v = np.linspace(-x_range, x_range, res)
        X, Y = np.meshgrid(x_v, y_v)
        Z = f_np(X, Y)

        fig = go.Figure()

        # 1. Origin "Net" / Reference Planes
        # Creating invisible-ish zero planes to solve the "wall" issue
        zero_line_style = dict(color="rgba(0,0,0,0.2)", width=2)
        fig.update_layout(scene=dict(
            xaxis=dict(showgrid=True, zeroline=True, zerolinethickness=4, zerolinecolor='red'),
            yaxis=dict(showgrid=True, zeroline=True, zerolinethickness=4, zerolinecolor='green'),
            zaxis=dict(showgrid=True, zeroline=True, zerolinethickness=4, zerolinecolor='blue'),
        ))

        # Main Surface (Becomes transparent in other modes)
        opac = 0.9 if mode == "Standard Surface" else 0.3
        fig.add_trace(go.Surface(z=Z, x=X, y=Y, opacity=opac, colorscale='Viridis', name='f(x,y)'))

        if mode == "Partial Derivative Mode":
            st.info("Showing Partial Derivative Visuals. Focus shifted from main surface.")
            # Calculate a sample trace for visualization
            dz_dx_np = sp.lambdify((x_s, y_s), df_dx, 'numpy')
            Z_deriv = dz_dx_np(X, Y)
            fig.add_trace(go.Surface(z=Z_deriv, x=X, y=Y, opacity=0.9, colorscale='Reds', name='df/dx'))

        elif mode == "Gradient Surface Mode":
            st.info("Showing Gradient Magnitude Surface. Focus shifted from main surface.")
            grad_mag = sp.sqrt(df_dx**2 + df_dy**2)
            grad_np = sp.lambdify((x_s, y_s), grad_mag, 'numpy')
            Z_grad = grad_np(X, Y)
            fig.add_trace(go.Surface(z=Z_grad, x=X, y=Y, opacity=0.9, colorscale='Magma', name='|Grad f|'))

        fig.update_traces(hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}")
        st.plotly_chart(fig, use_container_width=True)
        
        st.latex(f"f(x,y) = {sp.latex(f_s)}")
        st.latex(f"\\nabla f = \\langle {sp.latex(df_dx)}, {sp.latex(df_dy)} \\rangle")

    except Exception as e:
        st.error(f"Input Error: {e}")

# ---------------------------------------------------------
# PAGE 3: REAL-TIME LEVEL CURVES (Surface k)
# ---------------------------------------------------------
elif page == "Page 3: Real-Time Level Curves":
    st.title("🗺️ Page 3: Real-Time Level Surface Intersection")
    
    st.sidebar.subheader("Controls")
    user_input = st.sidebar.text_input("Function f(x,y):", "x**2 + y**2")
    k_val = st.sidebar.slider("Adjust Level k (z = k)", -10.0, 10.0, 1.0)
    show_main = st.sidebar.checkbox("Show Main Surface Transparently", value=True)

    try:
        f_s = smart_parse(user_input)
        f_np = sp.lambdify((x_s, y_s), f_s, 'numpy')
        
        x_v = np.linspace(-5, 5, 60)
        y_v = np.linspace(-5, 5, 60)
        X, Y = np.meshgrid(x_v, y_v)
        Z = f_np(X, Y)

        fig = go.Figure()

        # Transparent Main Surface
        if show_main:
            fig.add_trace(go.Surface(z=Z, x=X, y=Y, opacity=0.2, showscale=False))

        # Real-time Shifting k-plane (Surface k)
        K_plane = np.full_like(Z, k_val)
        fig.add_trace(go.Surface(z=K_plane, x=X, y=Y, opacity=0.5, colorscale='Blues', showscale=False, name=f'z={k_val}'))

        # Intersection / Level Curve (Contour on a 3D Plane)
        fig.add_trace(go.Contour(z=Z, x=x_v, y=y_v, 
                                 contours=dict(start=k_val, end=k_val, size=1),
                                 line=dict(width=5, color='red'),
                                 showscale=False))

        fig.update_layout(scene=dict(zaxis=dict(range=[-10, 10])))
        st.plotly_chart(fig, use_container_width=True)
        st.write(f"Level Curve at $z = {k_val}$")

    except Exception as e:
        st.error(f"Error: {e}")
