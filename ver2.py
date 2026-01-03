import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp

# Definition for Topic 1 (Required for CO1 Excellent grade)
st.title("MAT201: Function Visualization & Derivatives")
st.markdown("""
### Definition: Function of Two Variables
A function $z = f(x, y)$ assigns a unique real number to each pair $(x, y)$ in its domain. 
The set of all such points $(x, y, f(x, y))$ forms a surface in 3D space.
""")

# Sidebar for user input
user_func = st.sidebar.text_input("Enter function (e.g., x**2 + y**2):", "x**2 - y**2")

x_s, y_s = sp.symbols('x y')
try:
    f_s = sp.simplify(user_func)
    # Calculate Partial Derivatives (Topic 2 integration)
    df_dx = sp.diff(f_s, x_s)
    df_dy = sp.diff(f_s, y_s)

    # Math for the graph
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    f_np = sp.lambdify((x_s, y_s), f_s, 'numpy')
    Z = f_np(X, Y)

    # 3D Visualization with Hover Data
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_traces(hovertemplate="x: %{x}<br>y: %{y}<br>z: %{z}")
    st.plotly_chart(fig)

    st.write(f"**Partial Derivative w.r.t x:** ${sp.latex(df_dx)}$")
    st.write(f"**Partial Derivative w.r.t y:** ${sp.latex(df_dy)}$")
    
except Exception as e:
    st.error(f"Error: {e}")
