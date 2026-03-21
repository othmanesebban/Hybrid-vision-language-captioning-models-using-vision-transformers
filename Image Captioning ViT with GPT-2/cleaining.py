# cleanup.py
import os
import shutil
from pathlib import Path

# === CONFIG ===
USER_HOME = str(Path.home())
CACHE_DIRS = [
    os.path.join(USER_HOME, ".cache", "huggingface"),
    os.path.join(USER_HOME, ".cache", "torch"),
    os.path.join(USER_HOME, ".cache", "transformers"),
    os.path.join(USER_HOME, "AppData", "Local", "Temp"),  # Windows temp
]

CLEANED = []

print("🧹 Cleaning cache folders...\n")
for path in CACHE_DIRS:
    if os.path.exists(path):
        try:
            shutil.rmtree(path)
            CLEANED.append(path)
            print(f"✅ Removed: {path}")
        except Exception as e:
            print(f"❌ Failed to remove {path}: {e}")
    else:
        print(f"⚠️ Not found: {path}")

print("\n🧽 Done. Cleaned:")
for p in CLEANED:
    print(" -", p)

print("\n💡 Tip: You may also clear Downloads or Documents manually if needed.")
