# ---------------------------------------------------------
# PAGE 1: DEFINITIONS & EXAMPLES
# ---------------------------------------------------------
if page == "Page 1: Definitions & Examples":
    st.title("📖 Page 1: Functions of Two Variables")

    # --- NEW DEFINITION SECTION ---
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
        st.write("**Domain & Range:**")
        st.write("The set of all possible input pairs $(x, y)$ is the **Domain**, and the set of resulting values $z$ is the **Range**.")

    st.markdown("---")
    st.subheader("💡 Visualizing Common Surfaces")
    # ------------------------------

    cols = st.columns(2)
    for idx, (name, formula) in enumerate(presets.items()):
        with cols[idx % 2]:
            st.write(f"**{name}:** $f(x,y) = {formula}$")
            f_p = smart_parse(formula); f_n = sp.lambdify((x_s, y_s), f_p, 'numpy')
            px = np.linspace(-3, 3, 30); py = np.linspace(-3, 3, 30); PX, PY = np.meshgrid(px, py); PZ = f_n(PX, PY)
            fig = go.Figure(data=[go.Surface(z=PZ, x=PX, y=PY, showscale=False)])
            add_reference_planes(fig, [-3, 3], [-3, 3], [np.nanmin(PZ), np.nanmax(PZ)])
            st.plotly_chart(fig, use_container_width=True)
