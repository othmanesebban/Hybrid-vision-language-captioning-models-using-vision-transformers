# fix_numpy.py
import subprocess
import sys

try:
    import numpy as np
    print(f"✅ NumPy is installed: version {np.__version__}")
except ImportError:
    print("❌ NumPy is not installed. Attempting to install...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "--force-reinstall", "numpy"])
        import numpy as np
        print(f"✅ NumPy reinstalled successfully: version {np.__version__}")
    except Exception as e:
        print("❌ Failed to install NumPy:", e)
