import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
# Verified library and import
from st_mathlive import mathfield
from streamlit_plotly_events import plotly_events 

# --- APP CONFIGURATION ---
st.set_page_config(page_title="MAT201 Calculus Explorer", layout="wide")

if 'px0' not in st.session_state:
    st.session_state.px0 = 0.0
if 'py0' not in st.session_state:
    st.session_state.py0 = 0.0

def smart_parse(user_input):
    if not user_input: return None
    try:
        # Clean LaTeX output for SymPy
        clean = user_input.replace(r'\frac', '/').replace('{', '(').replace('}', ')')
        clean = clean.replace(r'\cdot', '*').replace('^', '**').replace(r'\pi', 'pi')
        for func in ['sin', 'cos', 'tan', 'sqrt', 'log', 'exp']:
            clean = clean.replace('\\' + func, func)
        
        transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))
        return parse_expr(clean, transformations=transformations)
    except:
        return None

# --- SIDEBAR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Page 1: Definitions & Examples", "Page 2: 3D Analysis"])

x_s, y_s = sp.symbols('x y')
presets = {"Linear": "x + y", "Rational": "5/(x**2 + y**2 + 1)", "Root": "sqrt(x**2 + y**2)", "Trigo": "sin(x)*cos(y)"}

if page == "Page 1: Definitions & Examples":
    st.title("📖 Page 1: Functions of Two Variables")
    st.info("A function of two variables assigns a unique $z$ to every $(x, y)$.")
    # ... (Simplified Page 1 for brevity, logic remains same)

elif page == "Page 2: 3D Analysis":
    st.title("🧊 Page 2: Tangent Lines & Gradient Analysis")
    
    # Mathematical Keyboard (MathLive)
    st.write("### Define your function $f(x,y)$")
    func_type = st.sidebar.selectbox("Presets:", ["Custom"] + list(presets.keys()))
    default_val = "x^2 - y^2" if func_type == "Custom" else presets[func_type].replace("**", "^")
    
    # Correct component call for streamlit-mathlive
    user_input, _ = mathfield(title="Enter Equation", value=default_val, key="math_editor")
    
    x_min, x_max = st.sidebar.slider("X Range", -10.0, 10.0, (-5.0, 5.0))
    y_min, y_max = st.sidebar.slider("Y Range", -10.0, 10.0, (-5.0, 5.0))
    
    show_dx = st.sidebar.checkbox("Show ∂f/∂x (Red)", value=True)
    show_dy = st.sidebar.checkbox("Show ∂f/∂y (Blue)", value=True)
    show_grad = st.sidebar.checkbox("Show Tangent Plane (Purple)", value=True)

    try:
        f_s = smart_parse(user_input)
        if f_s:
            df_dx = sp.diff(f_s, x_s); df_dy = sp.diff(f_s, y_s)
            f_np = sp.lambdify((x_s, y_s), f_s, 'numpy')

            x_v = np.linspace(x_min, x_max, 40); y_v = np.linspace(y_min, y_max, 40)
            X, Y = np.meshgrid(x_v, y_v); Z = f_np(X, Y)

            fig = go.Figure()
            fig.add_trace(go.Mesh3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), opacity=0.6, color='lightblue'))

            cx, cy = st.session_state.px0, st.session_state.py0
            z0 = float(f_s.subs({x_s: cx, y_s: cy}))
            sx = float(df_dx.subs({x_s: cx, y_s: cy}))
            sy = float(df_dy.subs({x_s: cx, y_s: cy}))

            if show_dx:
                tx = np.linspace(x_min, x_max, 40)
                fig.add_trace(go.Scatter3d(x=tx, y=[cy]*40, z=z0 + sx*(tx-cx), mode='lines', line=dict(color='red', width=8)))
            if show_dy:
                ty = np.linspace(y_min, y_max, 40)
                fig.add_trace(go.Scatter3d(x=[cx]*40, y=ty, z=z0 + sy*(ty-cy), mode='lines', line=dict(color='blue', width=8)))
            if show_grad:
                GZ = z0 + sx*(X - cx) + sy*(Y - cy)
                fig.add_trace(go.Surface(z=GZ, x=X, y=Y, opacity=0.3, colorscale=[[0, 'purple'], [1, 'purple']], showscale=False))

            # Point marker
            fig.add_trace(go.Scatter3d(x=[cx], y=[cy], z=[z0], mode='markers', marker=dict(size=8, color='black')))

            # Capture clicks to move the point
            selected = plotly_events(fig, click_event=True, key=f"plot_{user_input}")
            if selected:
                st.session_state.px0 = float(selected[0]['x'])
                st.session_state.py0 = float(selected[0]['y'])
                st.rerun()

            st.latex(f"\\nabla f({cx:.2f}, {cy:.2f}) = \\langle {sx:.2f}, {sy:.2f} \\rangle")
    except Exception as e:
        st.info("Please enter a valid function to begin.")
