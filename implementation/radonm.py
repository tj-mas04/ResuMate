import matplotlib.pyplot as plt

# Define project title and tasks with durations
project_title = "ResuMate - AI Resume Parser - Release Plan / Roadmap"

tasks = [
    ("Proposal & Planning", 1, 3),
    ("Data Collection & Resume Gathering", 3, 6),
    ("Data Cleaning & Preprocessing", 5, 8),
    ("Model Selection & Feature Engineering", 6, 9),
    ("Model Training & Optimization", 8, 11),
    ("Evaluation & Validation", 10, 13),
    ("Explainability & Interpretability", 12, 14),
    ("API Development & Integration", 13, 16),
    ("Frontend Development (UI)", 15, 17),
    ("Documentation & Reporting", 16, 18),
    ("Final Testing & Deployment", 17, 19),
]

colors = [
    "#3f51b5", "#7986cb", "#009688", "#4db6ac", "#43a047",
    "#7cb342", "#fbc02d", "#ffa000", "#fb8c00", "#ef5350", "#d32f2f"
]

# Create figure
fig, ax = plt.subplots(figsize=(14, 7))

# Plot tasks
for i, (task, start, end) in enumerate(tasks):
    ax.barh(i, end - start, left=start, color=colors[i], edgecolor='black')
    ax.text(start + 0.1, i, task, va='center', ha='left', color='white', fontsize=10, fontweight='bold')

# Set labels and titles
ax.set_yticks(range(len(tasks)))
ax.set_yticklabels([task[0] for task in tasks], fontsize=10)
ax.invert_yaxis()
ax.set_xticks(range(1, 21))
ax.set_xticklabels([f"W/{i}" for i in range(1, 21)], fontsize=9)
ax.set_xlabel("Weeks", fontsize=11)
ax.set_title(project_title, fontsize=15, weight='bold', pad=15)

# Add grid
ax.grid(True, axis='x', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()