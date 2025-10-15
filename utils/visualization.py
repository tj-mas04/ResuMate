"""
Visualization utilities for charts and plots.
"""
from math import pi
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st


def create_evaluation_plot(scores, ats_scores):
    """
    Create visualization comparing resume scores.
    
    Args:
        scores (dict): Resume names -> similarity scores
        ats_scores (dict): Resume names -> ATS scores
        
    Returns:
        str: Path to saved plot image, or None
    """
    labels = list(scores.keys())
    n = len(labels)
    
    if n == 0:
        st.info("No resumes to plot.")
        return None
    
    sim_values = [float(x) for x in scores.values()]
    ats_values = [float(x) for x in ats_scores.values()]
    
    # For fewer than 3 resumes, use bar chart
    if n < 3:
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(n)
        width = 0.35
        ax.bar(x - width/2, sim_values, width, label='Similarity (%)')
        ax.bar(x + width/2, ats_values, width, label='ATS Score')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylim(0, 100)
        ax.set_ylabel("Score")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.savefig("plot.png")
        return "plot.png"
    
    # Radar chart for 3+ resumes
    sim_vals = sim_values + sim_values[:1]
    ats_vals = ats_values + ats_values[:1]
    angles = [i * 2 * pi / n for i in range(n)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, sim_vals, 'o-', linewidth=2, label='Similarity (%)')
    ax.fill(angles, sim_vals, alpha=0.25)
    ax.plot(angles, ats_vals, 'o-', linewidth=2, label='ATS Score')
    ax.fill(angles, ats_vals, alpha=0.25)
    ax.set_thetagrids([a * 180 / pi for a in angles[:-1]], labels)
    ax.set_ylim(0, 100)
    ax.set_title("Resume Evaluation Radar Chart", y=1.05)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    plt.tight_layout()
    st.pyplot(fig)
    plt.savefig("plot.png")
    return "plot.png"
