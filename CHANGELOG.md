.venv/
venv/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.idea/
.vscode/
.DS_Store

# Data are intentionally not versioned.
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Experiment artefacts are generated locally.
results/runs/*
results/figures/*
results/summary.csv
!results/runs/.gitkeep
!results/figures/.gitkeep

*.h5
*.keras
*.ckpt
*.log
