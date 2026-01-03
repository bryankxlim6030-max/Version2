import streamlit as st
import numpy as np
import plotly.graph_objects as go
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor
# Make sure to run: pip install streamlit-plotly-events
from streamlit_plotly_events import plotly_events 

# --- APP CONFIGURATION ---
st.set_page_config(page_title="MAT201 Calculus Explorer", layout="wide")

# Initialize Session State for coordinates
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

def add_reference_planes(fig, x_r, y_r, z_r, show_z=True):
    fig.add_trace(go.Surface(x=[0, 0], y=[y_r[0], y_r[1]], z=np.array([[z_r[0], z_r[1]], [z_r[0], z_r[1]]]), 
                             opacity=0.1, colorscale=[[0, 'red'], [1, 'red']], showscale=False, hoverinfo='skip'))
    fig.add_trace(go.Surface(x=[x_r[0], x_r[1]], y=[0, 0], z=np.array([[z_r[0], z_r[1]], [z_r[0], z_r[1]]]).T, 
                             opacity=0.1, colorscale=[[0, 'blue'], [1, 'blue']], showscale=False, hoverinfo='skip'))
    if show_z:
        fig.add_trace(go.Surface(x=[x_r[0], x_r[1]], y=[y_r[0], y_r[1]], z=np.zeros((2,2)), 
                                 opacity=0.1, colorscale=[[0, 'yellow'], [1, 'yellow']], showscale=False, hoverinfo='skip'))

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Page 1: Definitions & Examples", "Page 2: 3D Analysis & Derivatives"])

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
        st.write("The set of all points $(x, y, z)$ forms a **Surface** in $\mathbb{R}^3$.")

    st.markdown("---")
    st.subheader("💡 Visualizing Common Surface Examples")

    # RESTORED EXAMPLES LOOP
    cols = st.columns(2)
    for idx, (name, formula) in enumerate(presets.items()):
        with cols[idx % 2]:
            st.write(f"**{name}:** $f(x,y) = {formula}$")
            f_p = smart_parse(formula); f_n = sp.lambdify((x_s, y_s), f_p, 'numpy')
            px = np.linspace(-3, 3, 30); py = np.linspace(-3, 3, 30); PX, PY = np.meshgrid(px, py); PZ = f_n(PX, PY)
            fig_eg = go.Figure(data=[go.Surface(z=PZ, x=PX, y=PY, showscale=False)])
            add_reference_planes(fig_eg, [-3, 3], [-3, 3], [np.nanmin(PZ), np.nanmax(PZ)])
            st.plotly_chart(fig_eg, use_container_width=True)

# ---------------------------------------------------------
# PAGE 2: 3D ANALYSIS (FIXED)
# ---------------------------------------------------------
elif page == "Page 2: 3D Analysis & Derivatives":
    st.title("🧊 Page 2: Tangent Lines & Gradient Analysis")
    st.caption("Click the surface to select a point, or use the sidebar for manual entry.")

    func_type = st.sidebar.selectbox("Function Mode:", ["Custom"] + list(presets.keys()))
    user_input = st.sidebar.text_input("Function f(x,y):", "x**2 - y**2") if func_type == "Custom" else presets[func_type]
    
    x_min, x_max = st.sidebar.slider("X Range", -10.0, 10.0, (-5.0, 5.0))
    y_min, y_max = st.sidebar.slider("Y Range", -10.0, 10.0, (-5.0, 5.0))
    mode = st.sidebar.radio("Mode:", ["Standard", "Analyse"])

    if mode == "Analyse":
        st.sidebar.subheader("Point Selection")
        # Direct Input updates session state
        st.session_state.px0 = st.sidebar.number_input("x coordinate", value=float(st.session_state.px0), step=0.1)
        st.session_state.py0 = st.sidebar.number_input("y coordinate", value=float(st.session_state.py0), step=0.1)
        
        show_dx = st.sidebar.checkbox("Show ∂f/∂x (Red)", value=True)
        show_dy = st.sidebar.checkbox("Show ∂f/∂y (Blue)", value=True)
        show_grad = st.sidebar.checkbox("Show Tangent Plane (Purple)", value=True)

    try:
        f_s = smart_parse(user_input)
        df_dx = sp.diff(f_s, x_s); df_dy = sp.diff(f_s, y_s)
        f_np = sp.lambdify((x_s, y_s), f_s, 'numpy')

        x_v = np.linspace(x_min, x_max, 50); y_v = np.linspace(y_min, y_max, 50)
        X, Y = np.meshgrid(x_v, y_v); Z = f_np(X, Y)

        fig = go.Figure()
        add_reference_planes(fig, [x_min, x_max], [y_min, y_max], [np.nanmin(Z), np.nanmax(Z)])
        
        main_opacity = 1.0 if mode == "Standard" else 0.4
        fig.add_trace(go.Surface(z=Z, x=X, y=Y, opacity=main_opacity, colorscale='Viridis', name='f(x,y)', hoverinfo='x+y+z'))

        if mode == "Analyse":
            curr_x, curr_y = st.session_state.px0, st.session_state.py0
            z0 = float(f_s.subs({x_s: curr_x, y_s: curr_y}))
            slope_x = float(df_dx.subs({x_s: curr_x, y_s: curr_y}))
            slope_y = float(df_dy.subs({x_s: curr_x, y_s: curr_y}))

            if show_dx:
                tx = np.linspace(x_min, x_max, 50)
                fig.add_trace(go.Scatter3d(x=tx, y=[curr_y]*50, z=z0 + slope_x*(tx - curr_x), mode='lines', line=dict(color='red', width=8), name='dx'))
            if show_dy:
                ty = np.linspace(y_min, y_max, 50)
                fig.add_trace(go.Scatter3d(x=[curr_x]*50, y=ty, z=z0 + slope_y*(ty - curr_y), mode='lines', line=dict(color='blue', width=8), name='dy'))
            if show_grad:
                GZ = z0 + slope_x*(X - curr_x) + slope_y*(Y - curr_y)
                fig.add_trace(go.Surface(z=GZ, x=X, y=Y, opacity=0.5, colorscale=[[0, 'purple'], [1, 'purple']], showscale=False, name='Plane'))
            
            fig.add_trace(go.Scatter3d(x=[curr_x], y=[curr_y], z=[z0], mode='markers', marker=dict(size=10, color='black'), name='Point'))

        # --- THE CLICK LOGIC (FIXED ARGUMENTS) ---
        if mode == "Analyse":
            # Removed use_container_width which was causing the error
            selected = plotly_events(fig, click_event=True, hover_event=False)
            
            if selected:
                st.session_state.px0 = float(selected[0]['x'])
                st.session_state.py0 = float(selected[0]['y'])
                st.rerun() 
        else:
            st.plotly_chart(fig, use_container_width=True)

        if mode == "Analyse":
            st.latex(f"f({st.session_state.px0:.2f}, {st.session_state.py0:.2f}) = {z0:.2f}")
            st.latex(r"\nabla f = \langle " + f"{slope_x:.2f}, {slope_y:.2f}" + r"\rangle")

    except Exception as e:
        st.error(f"Error: {e}")
