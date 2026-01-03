import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

# --- APP CONFIGURATION ---
st.set_page_config(page_title="MAT201 Calculus Explorer", layout="wide")

# --- SMART PARSER (Handles 2xy, x^2, etc.) ---
def smart_parse(user_input):
    try:
        clean_input = user_input.replace('^', '**')
        transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))
        return parse_expr(clean_input, transformations=transformations)
    except Exception as e:
        return None

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Page 1: Definitions & Examples", "Page 2: 3D Surface & Derivatives", "Page 3: Level Curves (f(x,y)=k)"])

# Shared Variables for logic
x_s, y_s = sp.symbols('x y')

# ---------------------------------------------------------
# PAGE 1: DEFINITIONS & EXAMPLES
# ---------------------------------------------------------
if page == "Page 1: Definitions & Examples":
    st.title("📖 Page 1: Understanding Functions of Two Variables")
    st.markdown("""
    ### Definition
    A **function of two variables** $z = f(x, y)$ is a rule that assigns to each ordered pair of real numbers $(x, y)$ in a domain $D$ a unique real number $z$.
    """)
    
    st.subheader("Visual Examples of Function Types")
    cols = st.columns(2)
    
    examples = [
        ("Linear", "x + y", "A flat tilted plane."),
        ("Rational", "1 / (x^2 + y^2 + 1)", "A bell-shaped curve."),
        ("Root", "sqrt(x^2 + y^2)", "A cone-shaped surface."),
        ("Trigonometric", "sin(x) * cos(y)", "A periodic 'egg-carton' surface.")
    ]
    
    for idx, (name, formula, desc) in enumerate(examples):
        with cols[idx % 2]:
            st.write(f"**{name} Function:** $f(x,y) = {formula}$")
            # Small preview plot
            f_p = smart_parse(formula)
            f_n = sp.lambdify((x_s, y_s), f_p, 'numpy')
            px = np.linspace(-3, 3, 30); py = np.linspace(-3, 3, 30)
            PX, PY = np.meshgrid(px, py)
            PZ = f_n(PX, PY)
            fig = go.Figure(data=[go.Surface(z=PZ, x=PX, y=PY, showscale=False)])
            fig.update_layout(height=300, margin=dict(l=0,r=0,b=0,t=0))
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# PAGE 2: 3D SURFACE & DERIVATIVES
# ---------------------------------------------------------
elif page == "Page 2: 3D Surface & Derivatives":
    st.title("🧊 Page 2: 3D Surface & Real-Time Derivatives")
    
    st.sidebar.subheader("Function Settings")
    preset = st.sidebar.selectbox("Choose a function:", ["Custom", "x + y", "1/(x**2 + y**2 + 1)", "sqrt(x**2 + y**2)", "sin(x)*cos(y)"])
    
    if preset == "Custom":
        user_input = st.sidebar.text_input("Enter function (e.g., 2xy, x^2):", "x**2 - y**2")
    else:
        user_input = preset

    # Range Controls (Individual X, Y, Z)
    st.sidebar.subheader("Axis Ranges")
    x_min, x_max = st.sidebar.slider("X Range", -10.0, 10.0, (-5.0, 5.0))
    y_min, y_max = st.sidebar.slider("Y Range", -10.0, 10.0, (-5.0, 5.0))
    z_auto = st.sidebar.checkbox("Auto-scale Z", value=True)
    z_range = [0, 0]
    if not z_auto:
        z_range = st.sidebar.slider("Z Range", -20.0, 20.0, (-5.0, 5.0))

    try:
        f_s = smart_parse(user_input)
        df_dx = sp.diff(f_s, x_s)
        df_dy = sp.diff(f_s, y_s)
        f_np = sp.lambdify((x_s, y_s), f_s, 'numpy')

        # Real-time data generation
        x_v = np.linspace(x_min, x_max, 50)
        y_v = np.linspace(y_min, y_max, 50)
        X, Y = np.meshgrid(x_v, y_v)
        Z = f_np(X, Y)

        # Plot with Real-time Hover Data
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
        fig.update_traces(hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}")
        
        if not z_auto:
            fig.update_layout(scene=dict(zaxis=dict(range=z_range)))
        
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Calculus Results")
        st.latex(f"\\frac{{\\partial f}}{{\\partial x}} = {sp.latex(df_dx)}")
        st.latex(f"\\frac{{\\partial f}}{{\\partial y}} = {sp.latex(df_dy)}")
        st.info("💡 Move your cursor over the graph to see coordinates and gradients.")

    except Exception as e:
        st.error(f"Error: {e}")

# ---------------------------------------------------------
# PAGE 3: LEVEL CURVES (f(x,y)=k)
# ---------------------------------------------------------
elif page == "Page 3: Level Curves (f(x,y)=k)":
    st.title("🗺️ Page 3: Level Curves & Contour Plots")
    
    st.sidebar.subheader("Contour Settings")
    user_input = st.sidebar.text_input("Function f(x,y):", "x**2 + y**2")
    k_val = st.sidebar.slider("Value of k (Level)", -10.0, 10.0, 0.0)
    
    try:
        f_s = smart_parse(user_input)
        f_np = sp.lambdify((x_s, y_s), f_s, 'numpy')
        
        x_v = np.linspace(-5, 5, 100)
        y_v = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x_v, y_v)
        Z = f_np(X, Y)

        # Contour Plot
        fig = go.Figure(data=go.Contour(z=Z, x=x_v, y=y_v, 
                                        contours=dict(start=k_val, end=k_val+0.1, size=0.1, showlabels=True),
                                        colorscale='Viridis'))
        
        fig.update_layout(title=f"Level Curve at f(x,y) = {k_val}")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown(f"**Current Level Equation:** ${sp.latex(f_s)} = {k_val}$")

    except Exception as e:
        st.error(f"Error: {e}")
