"""Pytest configuration for task-03-github-analyzer."""

import sys
from pathlib import Path

# Add task-03-github-analyzer directory to sys.path
TASK_DIR = Path(__file__).resolve().parent.parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))
