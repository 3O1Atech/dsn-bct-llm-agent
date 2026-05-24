"""
Backward-compatibility wrapper.
main.py calls scripts/seed_db.py on startup.
We forward to the new seed_chroma.py implementation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.seed_chroma import main

if __name__ == "__main__":
    main()
