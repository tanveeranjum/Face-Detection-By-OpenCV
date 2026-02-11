# check_env.py - Check your environment
import sys
import os
import subprocess

print("="*60)
print("🔍 CHECKING YOUR ENVIRONMENT")
print("="*60)

print(f"\nPython executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")

# Check if we're in virtual environment
in_venv = (
    hasattr(sys, 'real_prefix') or
    (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
    os.getenv('VIRTUAL_ENV') is not None
)

print(f"\nIn virtual environment: {in_venv}")

if os.getenv('VIRTUAL_ENV'):
    print(f"Virtual env path: {os.getenv('VIRTUAL_ENV')}")

print("\n📦 Checking installed packages...")

# List installed packages
try:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list"],
        capture_output=True,
        text=True
    )
    print(result.stdout)
except:
    print("Could not list packages")

print("\n🔧 Checking specific packages...")

packages_to_check = ['opencv-python', 'numpy', 'Pillow']

for package in packages_to_check:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"✅ {package} is installed")
        else:
            print(f"❌ {package} is NOT installed")
    except:
        print(f"⚠️ Could not check {package}")

print("\n" + "="*60)
print("To fix VS Code issues:")
print("1. Press Ctrl+Shift+P")
print("2. Type 'Python: Select Interpreter'")
print("3. Choose the one in your .venv folder")
print("="*60)