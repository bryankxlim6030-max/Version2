import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
from st_mathlive import mathfield
from streamlit_plotly_events import plotly_events 

# --- APP CONFIGURATION ---
st.set_page_config(page_title="MAT201 Calculus Explorer", layout="wide")

# Initialize Session State
if 'px0' not in st.session_state:
    st.session_state.px0 = 0.0
if 'py0' not in st.session_state:
    st.session_state.py0 = 0.0

def smart_parse(user_input):
    if not user_input: return None
    try:
        # Improved LaTeX cleaning for SymPy
        clean = user_input.replace(r'\frac', '/').replace('{', '(').replace('}', ')')
        clean = clean.replace(r'\cdot', '*').replace('^', '**').replace(r'\pi', 'pi')
        # Remove LaTeX backslashes from common functions
        for func in ['sin', 'cos', 'tan', 'sqrt', 'log', 'exp']:
            clean = clean.replace('\\' + func, func)
        
        transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))
        return parse_expr(clean, transformations=transformations)
    except:
        return None

def add_reference_planes(fig, x_r, y_r, z_r, show_z=True):
    # X=0 Plane (Red)
    fig.add_trace(go.Surface(x=[0, 0], y=[y_r[0], y_r[1]], z=np.array([[z_r[0], z_r[1]], [z_r[0], z_r[1]]]), 
                             opacity=0.1, colorscale=[[0, 'red'], [1, 'red']], showscale=False, hoverinfo='skip'))
    # Y=0 Plane (Blue)
    fig.add_trace(go.Surface(x=[x_r[0], x_r[1]], y=[0, 0], z=np.array([[z_r[0], z_r[1]], [z_r[0], z_r[1]]]).T, 
                             opacity=0.1, colorscale=[[0, 'blue'], [1, 'blue']], showscale=False, hoverinfo='skip'))
    # Z=0 Plane (Yellow)
    if show_z:
        fig.add_trace(go.Surface(x=[x_r[0], x_r[1]], y=[y_r[0], y_r[1]], z=np.zeros((2,2)), 
                                 opacity=0.1, colorscale=[[0, 'yellow'], [1, 'yellow']], showscale=False, hoverinfo='skip'))

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Page 1: Definitions & Examples", "Page 2: 3D Analysis"])

x_s, y_s = sp.symbols('x y')
presets = {"Linear": "x + y", "Rational": "5/(x**2 + y**2 + 1)", "Root": "sqrt(x**2 + y**2)", "Trigo": "sin(x)*cos(y)"}

# ---------------------------------------------------------
# PAGE 1: DEFINITIONS & EXAMPLES (RESTORED)
# ---------------------------------------------------------
if page == "Page 1: Definitions & Examples":
    st.title("📖 Page 1: Functions of Two Variables")
    st.info("### 📘 Mathematical Definition")
    st.markdown("""
    A **function of two variables** is a rule that assigns to each ordered pair of real numbers 
    $(x, y)$ in a set $D$ (the domain) a unique real number denoted by $f(x, y)$. 
    """)
    
    col_def1, col_def2 = st.columns(2)
    with col_def1:
        st.write("**General Form:**")
        st.latex(r"z = f(x, y)")
    with col_def2:
        st.write("**Geometric Interpretation:**")
        st.write("The set of all points $(x, y, z)$ such that $z = f(x, y)$ forms a **Surface** in 3D Space.")

    st.markdown("---")
    st.subheader("💡 Visualizing Common Surface Examples")

    cols = st.columns(2)
    for idx, (name, formula) in enumerate(presets.items()):
        with cols[idx % 2]:
            st.write(f"**{name}:** $f(x,y) = {formula}$")
            f_p = smart_parse(formula)
            if f_p:
                f_n = sp.lambdify((x_s, y_s), f_p, 'numpy')
                px = np.linspace(-3, 3, 30); py = np.linspace(-3, 3, 30)
                PX, PY = np.meshgrid(px, py); PZ = f_n(PX, PY)
                fig_eg = go.Figure(data=[go.Surface(z=PZ, x=PX, y=PY, showscale=False)])
                add_reference_planes(fig_eg, [-3, 3], [-3, 3], [float(np.nanmin(PZ)), float(np.nanmax(PZ))])
                st.plotly_chart(fig_eg, use_container_width=True)

# ---------------------------------------------------------
# PAGE 2: 3D ANALYSIS
# ---------------------------------------------------------
elif page == "Page 2: 3D Analysis":
    st.title("🧊 Page 2: Tangent Lines & Gradient Analysis")
    
    # 1. Math Keyboard Section (Pop-up style input)
    st.write("### ⌨️ Function Input")
    func_type = st.sidebar.selectbox("Presets:", ["Custom"] + list(presets.keys()))
    default_val = "x^2 - y^2" if func_type == "Custom" else presets[func_type].replace("**", "^")
    
    # This component provides the interactive math keyboard
    user_input, _ = mathfield(title="Click here to type your function:", value=default_val, key="math_editor")
    
    # Sidebar Controls
    x_min, x_max = st.sidebar.slider("X Range", -10.0, 10.0, (-5.0, 5.0))
    y_min, y_max = st.sidebar.slider("Y Range", -10.0, 10.0, (-5.0, 5.0))
    
    show_dx = st.sidebar.checkbox("Show ∂f/∂x (Red Line)", value=True)
    show_dy = st.sidebar.checkbox("Show ∂f/∂y (Blue Line)", value=True)
    show_grad = st.sidebar.checkbox("Show Tangent Plane (Purple)", value=True)

    try:
        f_s = smart_parse(user_input)
        if f_s:
            # Mathematical setup
            df_dx = sp.diff(f_s, x_s); df_dy = sp.diff(f_s, y_s)
            f_np = sp.lambdify((x_s, y_s), f_s, 'numpy')

            x_v = np.linspace(x_min, x_max, 40); y_v = np.linspace(y_min, y_max, 40)
            X, Y = np.meshgrid(x_v, y_v); Z = f_np(X, Y)

            fig = go.Figure()
            # Main surface using Mesh3d for better stability with plotly_events
            fig.add_trace(go.Mesh3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), 
                                    opacity=0.6, color='lightblue', name='Surface'))

            # Analysis at current point
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

            # Interactivity
            selected = plotly_events(fig, click_event=True, key=f"plot_{user_input}_{x_min}")
            if selected:
                st.session_state.px0 = float(selected[0]['x'])
                st.session_state.py0 = float(selected[0]['y'])
                st.rerun()

            st.markdown("---")
            st.latex(f"f({cx:.2f}, {cy:.2f}) = {z0:.2f} \\quad \\nabla f = \\langle {sx:.2f}, {sy:.2f} \\rangle")
            
    except Exception as e:
        st.info("Enter a valid mathematical function in the keyboard above to start visualization.")
