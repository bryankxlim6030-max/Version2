import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

# --- UI Header & Definition (CO1 Requirement) ---
st.set_page_config(page_title="MAT201 Function Visualizer", layout="wide")
st.title("MAT201: Interactive Function Explorer")

st.markdown("""
**Definition:** A function of two variables $z = f(x, y)$ maps pairs of inputs from the $xy$-plane to a specific height $z$. 
This app calculates the surface, its partial derivatives, and allows for natural math input.
""")

# --- Helper Function for Non-Coders ---
def smart_parse(user_input):
    # 1. Replace common non-python symbols
    clean_input = user_input.replace('^', '**')
    clean_input = clean_input.replace('e**', 'exp') # Handles e^x style
    
    # 2. Allow implicit multiplication (e.g., 2xy, (x+1)(y-1))
    # 'convert_xor' handles the ^ symbol if replace failed
    transformations = (standard_transformations + 
                       (implicit_multiplication_application, convert_xor))
    
    return parse_expr(clean_input, transformations=transformations)

# --- Sidebar ---
st.sidebar.header("Mathematical Input")
st.sidebar.info("You can type naturally: e.g., '2xy', 'sin(xy)', 'x^2 + y^2', or '(x+y)2'")

raw_input = st.sidebar.text_input("Enter your function f(x, y):", "2xy + sin(x)")

# --- Math Logic ---
x_s, y_s = sp.symbols('x y')

try:
    # Use our smart parser
    f_sympy = smart_parse(raw_input)
    
    # Calculate Derivatives for Topic 2
    df_dx = sp.diff(f_sympy, x_s)
    df_dy = sp.diff(f_sympy, y_s)

    # Numerical Conversion for Plotting
    f_func = sp.lambdify((x_s, y_s), f_sympy, 'numpy')
    
    x_val = np.linspace(-5, 5, 60)
    y_val = np.linspace(-5, 5, 60)
    X, Y = np.meshgrid(x_val, y_val)
    Z = f_func(X, Y)

    # --- 3D Visualization (CO1: Meaningful Visualization) ---
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Plasma')])
    
    # Custom Hover Info for Gradient/Coordinates
    fig.update_traces(
        hovertemplate="Point: (%{x:.2f}, %{y:.2f})<br>Height (z): %{z:.2f}<extra></extra>"
    )
    
    fig.update_layout(
        title=f"Surface Visualization of: {raw_input}",
        scene=dict(xaxis_title='X axis', yaxis_title='Y axis', zaxis_title='Z (Output)'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- Display Results ---
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Partial Derivative (∂f/∂x)")
        st.latex(sp.latex(df_dx))
    with c2:
        st.subheader("Partial Derivative (∂f/∂y)")
        st.latex(sp.latex(df_dy))

except Exception as e:
    st.error(f"Waiting for valid input... Error: {e}")
