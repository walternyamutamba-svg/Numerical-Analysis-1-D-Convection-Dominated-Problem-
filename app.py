import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="1D Convection–Diffusion Explorer",
    page_icon="📈",
    layout="wide",
)

# ------------------------------------------------------------
# Numerical model
# ------------------------------------------------------------
def exact_solution(x, epsilon, b, a=0.0, c=1.0, ua=0.0, uc=1.0):
    x = np.asarray(x, dtype=float)

    if abs(b) < 1e-14:
        return ua + (uc - ua) * (x - a) / (c - a)

    k = b / epsilon

    # Stable evaluation using exponentials shifted by c
    ex = np.exp(np.clip(k * (x - c), -700, 700))
    ea = np.exp(np.clip(k * (a - c), -700, 700))
    denom = 1.0 - ea

    if abs(denom) < 1e-14:
        return ua + (uc - ua) * (x - a) / (c - a)

    return ua + (uc - ua) * (ex - ea) / denom


def solve_central_difference(epsilon, b, a, c, ua, uc, N):
    h = (c - a) / N
    x = np.linspace(a, c, N + 1)
    m = N - 1

    u = np.zeros(N + 1)
    u[0], u[-1] = ua, uc
    if m == 0:
        return x, u

    A = np.zeros((m, m))
    rhs = np.zeros(m)

    lower = -epsilon / h**2 - b / (2.0 * h)
    diag = 2.0 * epsilon / h**2
    upper = -epsilon / h**2 + b / (2.0 * h)

    for i in range(m):
        A[i, i] = diag
        if i > 0:
            A[i, i - 1] = lower
        if i < m - 1:
            A[i, i + 1] = upper

    rhs[0] -= lower * ua
    rhs[-1] -= upper * uc
    u[1:N] = np.linalg.solve(A, rhs)
    return x, u


def solve_upwind_difference(epsilon, b, a, c, ua, uc, N):
    h = (c - a) / N
    x = np.linspace(a, c, N + 1)
    m = N - 1

    u = np.zeros(N + 1)
    u[0], u[-1] = ua, uc
    if m == 0:
        return x, u

    A = np.zeros((m, m))
    rhs = np.zeros(m)

    if b >= 0:
        lower = -epsilon / h**2 - b / h
        diag = 2.0 * epsilon / h**2 + b / h
        upper = -epsilon / h**2
    else:
        lower = -epsilon / h**2
        diag = 2.0 * epsilon / h**2 - b / h
        upper = -epsilon / h**2 + b / h

    for i in range(m):
        A[i, i] = diag
        if i > 0:
            A[i, i - 1] = lower
        if i < m - 1:
            A[i, i + 1] = upper

    rhs[0] -= lower * ua
    rhs[-1] -= upper * uc
    u[1:N] = np.linalg.solve(A, rhs)
    return x, u


def compute_errors(u_num, u_ex, h):
    err = u_num - u_ex
    l2 = np.sqrt(h * np.sum(err**2))
    linf = np.max(np.abs(err))
    return l2, linf


def compute_case(epsilon, b, a, c, ua, uc, N):
    x = np.linspace(a, c, N + 1)
    h = (c - a) / N
    u_exact = exact_solution(x, epsilon, b, a, c, ua, uc)
    _, u_central = solve_central_difference(epsilon, b, a, c, ua, uc, N)
    _, u_upwind = solve_upwind_difference(epsilon, b, a, c, ua, uc, N)

    l2_c, linf_c = compute_errors(u_central, u_exact, h)
    l2_u, linf_u = compute_errors(u_upwind, u_exact, h)

    return {
        "x": x,
        "h": h,
        "u_exact": u_exact,
        "u_central": u_central,
        "u_upwind": u_upwind,
        "err_c": np.abs(u_central - u_exact),
        "err_u": np.abs(u_upwind - u_exact),
        "l2_c": l2_c,
        "linf_c": linf_c,
        "l2_u": l2_u,
        "linf_u": linf_u,
        "peclet": abs(b) * h / (2.0 * epsilon) if epsilon > 0 else np.inf,
    }


@st.cache_data(show_spinner=False)
def epsilon_sweep(eps_values, b, a, c, ua, uc, N):
    rows = []
    frames = []
    for eps in eps_values:
        case = compute_case(float(eps), b, a, c, ua, uc, N)
        rows.append(
            {
                "epsilon": float(eps),
                "L2 Central": case["l2_c"],
                "Linf Central": case["linf_c"],
                "L2 Upwind": case["l2_u"],
                "Linf Upwind": case["linf_u"],
                "Cell Peclet": case["peclet"],
            }
        )
        frames.append(case)
    return pd.DataFrame(rows), frames


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------
st.title("1D Convection–Diffusion Streamlit App")
st.markdown(
    "Explore how the solution shape, numerical oscillations, and error change as the diffusion coefficient $\\epsilon$ varies."
)

with st.sidebar:
    st.header("Controls")
    epsilon = st.slider("Current epsilon", 0.0005, 0.2, 0.01, 0.0005, format="%.4f")
    b = st.slider("Convection coefficient b", -3.0, 3.0, 1.0, 0.1)
    c = st.slider("Right endpoint c", 0.5, 3.0, 1.0, 0.1)
    N = st.slider("Number of subintervals N", 10, 300, 80, 5)

    st.divider()
    st.subheader("Animation sweep")
    eps_min = st.number_input("epsilon min", min_value=0.0005, max_value=0.2, value=0.001, step=0.0005, format="%.4f")
    eps_max = st.number_input("epsilon max", min_value=0.0005, max_value=0.2, value=0.08, step=0.0005, format="%.4f")
    n_frames = st.slider("Number of animation frames", 5, 80, 20, 1)

if eps_min >= eps_max:
    st.error("epsilon min must be smaller than epsilon max.")
    st.stop()

a, ua, uc = 0.0, 0.0, 1.0
case = compute_case(epsilon, b, a, c, ua, uc, N)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("h", f"{case['h']:.4f}")
col2.metric("Cell Peclet", f"{case['peclet']:.3f}")
col3.metric("Central L2", f"{case['l2_c']:.3e}")
col4.metric("Upwind L2", f"{case['l2_u']:.3e}")
col5.metric("epsilon", f"{epsilon:.4f}")

if case["peclet"] > 1:
    st.warning(
        "Cell Peclet number is greater than 1. Central differencing may show non-physical oscillations in the convection-dominated regime."
    )
else:
    st.success("Cell Peclet number is at most 1. Central differencing is in a safer regime.")

# ------------------------------------------------------------
# Figure 1: Current solution + current error
# ------------------------------------------------------------
fig_current = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Solution behavior", "Pointwise absolute error"),
)

x = case["x"]
fig_current.add_trace(go.Scatter(x=x, y=case["u_exact"], mode="lines", name="Exact"), row=1, col=1)
fig_current.add_trace(go.Scatter(x=x, y=case["u_central"], mode="lines+markers", name="Central"), row=1, col=1)
fig_current.add_trace(go.Scatter(x=x, y=case["u_upwind"], mode="lines+markers", name="Upwind"), row=1, col=1)

fig_current.add_trace(go.Scatter(x=x, y=case["err_c"], mode="lines+markers", name="|Central-Exact|"), row=1, col=2)
fig_current.add_trace(go.Scatter(x=x, y=case["err_u"], mode="lines+markers", name="|Upwind-Exact|"), row=1, col=2)

fig_current.update_xaxes(title_text="x", row=1, col=1)
fig_current.update_xaxes(title_text="x", row=1, col=2)
fig_current.update_yaxes(title_text="u(x)", row=1, col=1)
fig_current.update_yaxes(title_text="Absolute error", row=1, col=2)
fig_current.update_layout(height=480, legend_title_text="Method")

st.plotly_chart(fig_current, use_container_width=True)

# ------------------------------------------------------------
# Sweep data
# ------------------------------------------------------------
eps_values = np.linspace(eps_min, eps_max, n_frames)
df_err, frames = epsilon_sweep(tuple(np.round(eps_values, 10)), b, a, c, ua, uc, N)

# ------------------------------------------------------------
# Figure 2: Animated behavior with epsilon
# ------------------------------------------------------------
initial = frames[0]
fig_anim = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Animated solutions over epsilon", "Animated pointwise error over epsilon"),
)

fig_anim.add_trace(go.Scatter(x=initial["x"], y=initial["u_exact"], mode="lines", name="Exact"), row=1, col=1)
fig_anim.add_trace(go.Scatter(x=initial["x"], y=initial["u_central"], mode="lines+markers", name="Central"), row=1, col=1)
fig_anim.add_trace(go.Scatter(x=initial["x"], y=initial["u_upwind"], mode="lines+markers", name="Upwind"), row=1, col=1)

fig_anim.add_trace(go.Scatter(x=initial["x"], y=initial["err_c"], mode="lines+markers", name="|Central-Exact|"), row=1, col=2)
fig_anim.add_trace(go.Scatter(x=initial["x"], y=initial["err_u"], mode="lines+markers", name="|Upwind-Exact|"), row=1, col=2)

plot_frames = []
for frame_case in frames:
    eps = frame_case["peclet"]
    plot_frames.append(
        go.Frame(
            name=f"{frame_case['l2_c']:.8f}",
            data=[
                go.Scatter(x=frame_case["x"], y=frame_case["u_exact"]),
                go.Scatter(x=frame_case["x"], y=frame_case["u_central"]),
                go.Scatter(x=frame_case["x"], y=frame_case["u_upwind"]),
                go.Scatter(x=frame_case["x"], y=frame_case["err_c"]),
                go.Scatter(x=frame_case["x"], y=frame_case["err_u"]),
            ],
            traces=[0, 1, 2, 3, 4],
            layout=go.Layout(
                title_text=(
                    f"Epsilon sweep animation | epsilon = {frame_case['h'] * frame_case['peclet'] * 2 / abs(b) if abs(b) > 1e-14 else eps_min:.5f}"
                    if abs(b) > 1e-14 else "Epsilon sweep animation"
                )
            ),
        )
    )

# Rebuild titles using actual epsilon labels
plot_frames = []
for eps_val, frame_case in zip(eps_values, frames):
    plot_frames.append(
        go.Frame(
            name=f"{eps_val:.5f}",
            data=[
                go.Scatter(x=frame_case["x"], y=frame_case["u_exact"]),
                go.Scatter(x=frame_case["x"], y=frame_case["u_central"]),
                go.Scatter(x=frame_case["x"], y=frame_case["u_upwind"]),
                go.Scatter(x=frame_case["x"], y=frame_case["err_c"]),
                go.Scatter(x=frame_case["x"], y=frame_case["err_u"]),
            ],
            traces=[0, 1, 2, 3, 4],
            layout=go.Layout(title_text=f"Epsilon sweep animation | epsilon = {eps_val:.5f}")
        )
    )

fig_anim.frames = plot_frames
fig_anim.update_xaxes(title_text="x", row=1, col=1)
fig_anim.update_xaxes(title_text="x", row=1, col=2)
fig_anim.update_yaxes(title_text="u(x)", row=1, col=1)
fig_anim.update_yaxes(title_text="Absolute error", row=1, col=2)
fig_anim.update_layout(
    height=520,
    title=f"Epsilon sweep animation | epsilon = {eps_values[0]:.5f}",
    updatemenus=[
        {
            "type": "buttons",
            "direction": "left",
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True}],
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                },
            ],
            "pad": {"r": 10, "t": 10},
            "x": 0.05,
            "y": 1.12,
        }
    ],
    sliders=[
        {
            "active": 0,
            "currentvalue": {"prefix": "epsilon = "},
            "pad": {"t": 45},
            "steps": [
                {
                    "label": f"{eps_val:.4f}",
                    "method": "animate",
                    "args": [[f"{eps_val:.5f}"], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                }
                for eps_val in eps_values
            ],
        }
    ],
)

st.plotly_chart(fig_anim, use_container_width=True)

# ------------------------------------------------------------
# Figure 3: Error change with epsilon
# ------------------------------------------------------------
fig_error = go.Figure()
fig_error.add_trace(go.Scatter(x=df_err["epsilon"], y=df_err["L2 Central"], mode="lines+markers", name="Central L2"))
fig_error.add_trace(go.Scatter(x=df_err["epsilon"], y=df_err["L2 Upwind"], mode="lines+markers", name="Upwind L2"))
fig_error.add_trace(go.Scatter(x=df_err["epsilon"], y=df_err["Linf Central"], mode="lines+markers", name="Central Linf"))
fig_error.add_trace(go.Scatter(x=df_err["epsilon"], y=df_err["Linf Upwind"], mode="lines+markers", name="Upwind Linf"))

# mark current epsilon if it lies inside the sweep interval
if eps_min <= epsilon <= eps_max:
    current_row = {
        "epsilon": epsilon,
        "Central L2": case["l2_c"],
        "Upwind L2": case["l2_u"],
    }
    fig_error.add_trace(
        go.Scatter(
            x=[current_row["epsilon"]],
            y=[current_row["Central L2"]],
            mode="markers",
            name="Current epsilon (Central L2)",
            marker=dict(size=12, symbol="diamond"),
        )
    )
    fig_error.add_trace(
        go.Scatter(
            x=[current_row["epsilon"]],
            y=[current_row["Upwind L2"]],
            mode="markers",
            name="Current epsilon (Upwind L2)",
            marker=dict(size=12, symbol="diamond"),
        )
    )

fig_error.update_layout(
    title="How the error changes with epsilon",
    xaxis_title="epsilon",
    yaxis_title="Error",
    height=500,
)

st.plotly_chart(fig_error, use_container_width=True)

with st.expander("See numerical error table"):
    st.dataframe(df_err, use_container_width=True)

st.markdown(
    """
### Interpretation guide
- Small **epsilon** means weaker diffusion and a stronger boundary layer near the outflow boundary.
- When the **cell Peclet number** is large, the **central difference method** can oscillate.
- The **upwind method** is usually more stable, but it may be more diffusive and slightly smear the sharp layer.
- The animation helps you see how both the **solution shape** and the **error curves** react as epsilon changes.
"""
)
