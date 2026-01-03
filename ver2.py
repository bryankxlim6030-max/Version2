import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

# --- 1. KEEPING THE DEFINITION (Required for CO1) ---
st.set_page_config(page_title="MAT201 Function Visualizer", layout="wide")
st.title("MAT201: Interactive Function Explorer")

st.markdown("""
### Definition: Function of Two Variables
A function $z = f(x, y)$ assigns a unique real number to each pair $(x, y)$ in its domain. 
The set of all such points $(x, y, f(x, y))$ forms a surface in 3D space.
""")

# --- 2. NEW IMPROVEMENT: Smart Parser (Handles 2xy, x^2, etc.) ---
def smart_parse(user_input):
    # Standardizes common non-python math symbols
    clean_input = user_input.replace('^', '**')
    # Transformations allow '2xy' to become '2*x*y' and handles implicit brackets
    transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))
    return parse_expr(clean_input, transformations=transformations)

# --- 3. KEEPING THE SIDEBAR & INPUT ---
st.sidebar.header("User Input")
st.sidebar.info("Style: '2xy', 'x^2+y^2', 'sin(x)cos(y)'")
user_func = st.sidebar.text_input("Input your function f(x, y):", "x**2 - y**2")

x_s, y_s = sp.symbols('x y')

try:
    # Applying the smart parser to the user input
    f_s = smart_parse(user_func)
    
    # --- 4. KEEPING PARTIAL DERIVATIVES (Topic 2 integration) ---
    df_dx = sp.diff(f_s, x_s)
    df_dy = sp.diff(f_s, y_s)

    # --- 5. KEEPING THE MATH FOR THE GRAPH ---
    x = np.linspace(-5, 5, 60)
    y = np.linspace(-5, 5, 60)
    X, Y = np.meshgrid(x, y)
    f_np = sp.lambdify((x_s, y_s), f_s, 'numpy')
    Z = f_np(X, Y)

    # --- 6. KEEPING 3D VISUALIZATION WITH HOVER DATA ---
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
    
    # Hover template shows coordinates and height as the cursor moves
    fig.update_traces(
        hovertemplate="x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<extra></extra>"
    )
    
    fig.update_layout(
        title=f"Surface Visualization of z = {user_func}",
        scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # --- 7. KEEPING THE CALCULATED RESULTS DISPLAY ---
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Partial Derivative w.r.t x:**")
        st.latex(sp.latex(df_dx))
    with col2:
        st.write("**Partial Derivative w.r.t y:**")
        st.latex(sp.latex(df_dy))
    
except Exception as e:
    st.error(f"Waiting for valid math input... (Error: {e})")
