import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

# 1. Definition (Required for CO1 Fundamental Concepts)
st.title("MAT201: Interactive Function Explorer")
st.markdown("""
### Definition: Function of Two Variables
A function $z = f(x, y)$ assigns a unique real number to each pair $(x, y)$ in its domain $D$. 
The set of all points $(x, y, z)$ satisfying $z = f(x, y)$ forms a **surface** in 3D space.
""")

# Sidebar Improvements
st.sidebar.header("Settings & Input")

# Feature: Implicit Multiplication Handler
def parse_math(user_input):
    transformations = (standard_transformations + (implicit_multiplication_application,))
    return parse_expr(user_input, transformations=transformations)

# Feature: Preset Examples for "Two Different Complexities"
example = st.sidebar.selectbox("Choose a preset or type your own:", 
                                ["Custom", "x^2 + y^2", "sin(x) * cos(y)", "exp(-(x^2 + y^2))"])

if example == "Custom":
    user_input = st.sidebar.text_input("Enter function (e.g., 2xy, x^2, sin(x)):", "x**2 - y**2")
else:
    user_input = example

# Feature: Sliders for Range Control
res = st.sidebar.slider("Graph Resolution", 20, 100, 50)
xy_limit = st.sidebar.slider("X & Y Range", 1, 10, 5)

x_s, y_s = sp.symbols('x y')

try:
    # Handle the flexible input style (e.g., 2xy -> 2*x*y)
    f_s = parse_math(user_input.replace('^', '**'))
    
    # Calculate Partial Derivatives (Topic 2)
    df_dx = sp.diff(f_s, x_s)
    df_dy = sp.diff(f_s, y_s)

    # Prepare Numerical Data
    x_vec = np.linspace(-xy_limit, xy_limit, res)
    y_vec = np.linspace(-xy_limit, xy_limit, res)
    X, Y = np.meshgrid(x_vec, y_vec)
    f_np = sp.lambdify((x_s, y_s), f_s, 'numpy')
    Z = f_np(X, Y)

    # 2. Interactive 3D Visualization (CO1 Cognitive Application)
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    
    # Feature: Enhanced Hover Labels
    fig.update_traces(
        hovertemplate="<b>Coord:</b> (%{x:.2f}, %{y:.2f})<br><b>Height (z):</b> %{z:.2f}<extra></extra>"
    )
    
    fig.update_layout(title=f"Surface of z = {user_input}", autosize=True)
    st.plotly_chart(fig, use_container_width=True)

    # Results Display
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Partial Derivative w.r.t $x$:**")
        st.latex(sp.latex(df_dx))
    with col2:
        st.write("**Partial Derivative w.r.t $y$:**")
        st.latex(sp.latex(df_dy))

except Exception as e:
    st.error(f"Input Error: Please check your syntax. (Error: {e})")
