import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
from streamlit_plotly_events import plotly_events 

# --- APP CONFIGURATION ---
st.set_page_config(page_title="MAT201 Calculus Explorer", layout="wide")

# Initialize Session State
if 'px0' not in st.session_state:
    st.session_state.px0 = 0.0
if 'py0' not in st.session_state:
    st.session_state.py0 = 0.0

def smart_parse(user_input):
    try:
        clean_input = user_input.replace('^', '**')
        transformations = (standard_transformations + (implicit_multiplication_application, convert_xor))
        return parse_expr(clean_input, transformations=transformations)
    except:
        return None

# --- SIDEBAR ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Page 1: Definitions & Examples", "Page 2: 3D Analysis & Derivatives"])

x_s, y_s = sp.symbols('x y')
presets = {"Linear": "x + y", "Rational": "5/(x**2 + y**2 + 1)", "Root": "sqrt(x**2 + y**2)", "Trigo": "sin(x)*cos(y)"}

# --- PAGE 1 ---
if page == "Page 1: Definitions & Examples":
    st.title("📖 Page 1: Functions of Two Variables")
    cols = st.columns(2)
    for idx, (name, formula) in enumerate(presets.items()):
        with cols[idx % 2]:
            st.write(f"**{name}:** $f(x,y) = {formula}$")
            f_p = smart_parse(formula)
            if f_p:
                f_n = sp.lambdify((x_s, y_s), f_p, 'numpy')
                px = np.linspace(-3, 3, 20); py = np.linspace(-3, 3, 20)
                PX, PY = np.meshgrid(px, py); PZ = f_n(PX, PY)
                fig_eg = go.Figure(data=[go.Surface(z=PZ, x=PX, y=PY, showscale=False)])
                st.plotly_chart(fig_eg, use_container_width=True)

# --- PAGE 2 ---
elif page == "Page 2: 3D Analysis & Derivatives":
    st.title("🧊 Page 2: Tangent Lines & Gradient Analysis")
    
    func_type = st.sidebar.selectbox("Function Mode:", ["Custom"] + list(presets.keys()))
    user_input = st.sidebar.text_input("Function f(x,y):", "x**2 - y**2") if func_type == "Custom" else presets[func_type]
    x_min, x_max = st.sidebar.slider("X Range", -10.0, 10.0, (-5.0, 5.0))
    y_min, y_max = st.sidebar.slider("Y Range", -10.0, 10.0, (-5.0, 5.0))
    mode = st.sidebar.radio("Mode:", ["Standard", "Analyse"])

    try:
        f_s = smart_parse(user_input)
        if f_s:
            df_dx = sp.diff(f_s, x_s); df_dy = sp.diff(f_s, y_s)
            f_np = sp.lambdify((x_s, y_s), f_s, 'numpy')
            
            # Generate mesh
            x_v = np.linspace(x_min, x_max, 40); y_v = np.linspace(y_min, y_max, 40)
            X, Y = np.meshgrid(x_v, y_v); Z = f_np(X, Y)

            fig = go.Figure()

            if mode == "Standard":
                fig.add_trace(go.Surface(z=Z, x=X, y=Y, colorscale='Viridis'))
                st.plotly_chart(fig, use_container_width=True)
            
            else:
                # --- ANALYSE MODE: STABILITY FIX ---
                st.sidebar.subheader("Point Selection")
                st.session_state.px0 = st.sidebar.number_input("x", value=float(st.session_state.px0), step=0.1)
                st.session_state.py0 = st.sidebar.number_input("y", value=float(st.session_state.py0), step=0.1)
                
                curr_x, curr_y = st.session_state.px0, st.session_state.py0
                z0 = float(f_s.subs({x_s: curr_x, y_s: curr_y}))
                sx = float(df_dx.subs({x_s: curr_x, y_s: curr_y}))
                sy = float(df_dy.subs({x_s: curr_x, y_s: curr_y}))

                # 1. Main Surface (Using Mesh3d for better interaction stability)
                fig.add_trace(go.Mesh3d(x=X.flatten(), y=Y.flatten(), z=Z.flatten(), 
                                        opacity=0.7, color='lightgray', name='Surface'))
                
                # 2. Tangent Lines
                tx = np.linspace(x_min, x_max, 40)
                fig.add_trace(go.Scatter3d(x=tx, y=[curr_y]*40, z=z0 + sx*(tx-curr_x), 
                                         mode='lines', line=dict(color='red', width=8), name='df/dx'))
                
                ty = np.linspace(y_min, y_max, 40)
                fig.add_trace(go.Scatter3d(x=[curr_x]*40, y=ty, z=z0 + sy*(ty-curr_y), 
                                         mode='lines', line=dict(color='blue', width=8), name='df/dy'))
                
                # 3. Target Point
                fig.add_trace(go.Scatter3d(x=[curr_x], y=[curr_y], z=[z0], 
                                         mode='markers', marker=dict(size=10, color='black')))

                # UNIQUE KEY based on function to force refresh
                selected = plotly_events(fig, click_event=True, key=f"fixed_plot_{user_input}")
                
                if selected:
                    st.session_state.px0 = float(selected[0]['x'])
                    st.session_state.py0 = float(selected[0]['y'])
                    st.rerun()

                st.latex(f"f({curr_x:.2f}, {curr_y:.2f}) = {z0:.2f}")
                st.latex(r"\nabla f = \langle" + f"{sx:.2f}, {sy:.2f}" + r"\rangle")
                
                if st.button("🔄 Force Refresh Graph"):
                    st.rerun()

    except Exception as e:
        st.error(f"Error: {e}")
