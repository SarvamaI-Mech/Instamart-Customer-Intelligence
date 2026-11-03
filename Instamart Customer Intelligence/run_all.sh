#!/usr/bin/env bash
# Runs the whole Copilot end to end from the repository root (after the virtual environment is active).
set -euo pipefail
python generate_data.py
python part_a_clean.py
python validate_checkpoints.py
python part_a_sql.py
python part_a_eda.py
python part_b_segments.py
python part_b_lapse.py
python part_c_triage.py
python part_d_rag.py
python part_d_agent.py
