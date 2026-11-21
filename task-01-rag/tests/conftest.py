import sys
from pathlib import Path

# Add task-01-rag directory to sys.path for test resolution
TASK_DIR = Path(__file__).resolve().parent.parent
if str(TASK_DIR) not in sys.path:
    sys.path.insert(0, str(TASK_DIR))
