from __future__ import annotations
from math import pi
import io
import hashlib
import numpy as np
import streamlit as st
import plotly.graph_objects as go


def save_plot(scores: dict[str, float], ats: dict[str, float], plot_type: str = "Auto (recommended)", display: bool = True, return_bytes: bool = False, **kwargs):
    """Render interactive Plotly charts for resume evaluation and optionally export a PNG.

    If `return_bytes` is True, the function will attempt to return PNG bytes (in-memory)
    using Plotly's image export (kaleido). If False it will try to write `plot.png` to disk
    and return the path string when successful. If export fails, returns None.
    """
    labels = list(scores.keys())
    n = len(labels)

    if n == 0:
        st.info("No resumes to plot.")
        return None

    sim_values = [float(x) for x in scores.values()]
    ats_values = [float(x) for x in ats.values()]

    pt = (plot_type or "Auto (recommended)").lower()

    # stable key base for Streamlit plotly_chart element
    base_src = plot_type + "|" + str(len(labels)) + "|" + ";".join(labels)
    if display:
        # append a render sequence so multiple renderings in the same run get unique keys
        seq = st.session_state.get("_plotly_render_seq", 0)
        st.session_state["_plotly_render_seq"] = seq + 1
        key_src = f"{base_src}|disp{seq}"
    else:
        # static-save key (won't render a chart) — keep different from display keys
        key_src = f"{base_src}|save"

    chart_key = "plot_" + hashlib.sha1(key_src.encode()).hexdigest()[:12]

    def try_write(fig):
        # Prefer in-memory bytes when requested (uses kaleido)
        if return_bytes:
            try:
                # Primary: fig.to_image() returns bytes
                return fig.to_image(format="png")
            except Exception as e_to:
                # Fallback: try writing into a BytesIO buffer
                try:
                    buf = io.BytesIO()
                    fig.write_image(buf, format="png")
                    return buf.getvalue()
                except Exception as e_buf:
                    # Save exception details for UI diagnostics
                    st.session_state["_plot_export_error"] = f"to_image error: {e_to}; write_image buffer error: {e_buf}"
                    return None
        try:
            fig.write_image("plot.png")
            return "plot.png"
        except Exception as e_file:
            st.session_state["_plot_export_error"] = f"write_image file error: {e_file}"
            return None

    # Horizontal grouped bar
    def do_horizontal():
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sim_values, y=labels, orientation="h", name="Similarity (%)", marker_color="#3182ce", text=[f"{v:.1f}%" for v in sim_values], textposition="outside"))
        fig.add_trace(go.Bar(x=ats_values, y=labels, orientation="h", name="ATS Score", marker_color="#2b6cb0", text=[f"{v:.1f}" for v in ats_values], textposition="outside"))
        fig.update_layout(barmode="group", xaxis=dict(range=[0, 100], title="Score"), height=max(300, 80 * n), margin=dict(l=220 if max(len(l) for l in labels) > 30 else 120, r=40, t=60, b=60))
        if display:
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
        return try_write(fig)

    # Lollipop: thin line + marker
    def do_lollipop():
        fig = go.Figure()
        fig.add_trace(go.Bar(x=sim_values, y=labels, orientation="h", marker_color="#b6d0f5", name="Similarity (line)", opacity=0.6))
        fig.add_trace(go.Bar(x=ats_values, y=labels, orientation="h", marker_color="#9eb6df", name="ATS (line)", opacity=0.6))
        fig.add_trace(go.Scatter(x=sim_values, y=labels, mode="markers+text", marker=dict(color="#3182ce", size=10), name="Similarity", text=[f"{v:.1f}%" for v in sim_values], textposition="middle right"))
        fig.add_trace(go.Scatter(x=ats_values, y=labels, mode="markers+text", marker=dict(color="#2b6cb0", size=10), name="ATS", text=[f"{v:.1f}" for v in ats_values], textposition="middle right"))
        fig.update_layout(barmode="overlay", xaxis=dict(range=[0, 100], title="Score"), height=max(300, 80 * n), margin=dict(l=220 if max(len(l) for l in labels) > 30 else 120, r=40, t=60, b=60))
        if display:
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
        return try_write(fig)

    # Heatmap
    def do_heatmap():
        z = [sim_values, ats_values]
        y = ["Similarity (%)", "ATS Score"]
        fig = go.Figure(data=go.Heatmap(z=z, x=labels, y=y, colorscale="YlGnBu", zmin=0, zmax=100, hovertemplate="%{y}<br>%{x}: %{z}<extra></extra>"))
        fig.update_layout(height=300, margin=dict(l=120, r=40, t=40, b=120))
        if display:
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
        return try_write(fig)

    # Radar / polar chart with annotations
    def do_radar():
        r_sim = sim_values + sim_values[:1]
        r_ats = ats_values + ats_values[:1]
        theta = labels + [labels[0]]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=r_sim, theta=theta, fill="toself", name="Similarity (%)", marker=dict(color="#3182ce"), hovertemplate="%{theta}: %{r}<extra></extra>"))
        fig.add_trace(go.Scatterpolar(r=r_ats, theta=theta, fill="toself", name="ATS Score", marker=dict(color="#2b6cb0"), hovertemplate="%{theta}: %{r}<extra></extra>"))
        # annotation traces (text near each point)
        fig.add_trace(go.Scatterpolar(r=r_sim, theta=theta, mode="markers+text", marker=dict(size=6, color="#3182ce"), text=[f"{v:.1f}%" for v in r_sim], textposition="top center", showlegend=False))
        fig.add_trace(go.Scatterpolar(r=r_ats, theta=theta, mode="markers+text", marker=dict(size=6, color="#2b6cb0"), text=[f"{v:.1f}" for v in r_ats], textposition="bottom center", showlegend=False))

        fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])), title=dict(text="Resume Evaluation Radar Chart", x=0.5), height=600, margin=dict(t=80, b=40))
        if display:
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
        return try_write(fig)

    # choose plotting function
    if "horizontal" in pt:
        return do_horizontal()
    if "lollipop" in pt or "dot" in pt:
        return do_lollipop()
    if "heat" in pt:
        return do_heatmap()
    if "radar" in pt:
        return do_radar()

    # Auto: radar for >=3, horizontal otherwise
    if n < 3:
        return do_horizontal()
    return do_radar()
