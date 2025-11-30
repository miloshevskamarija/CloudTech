import sys
import os

# Absolute path to project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")

# Prepend src to sys.path so it overrides site-packages
sys.path.insert(0, SRC_PATH)
sys.path.insert(0, PROJECT_ROOT)