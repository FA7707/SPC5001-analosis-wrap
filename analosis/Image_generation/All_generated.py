import subprocess
import sys

for script in ["Arc.py", "Multiple.py", "Non_lens.py","Ring.py"]:
    print(f"Running {script}...")
    subprocess.run([sys.executable, script], cwd=".", check=True)

print(f"Running export.py...")
subprocess.run([sys.executable, 'export.py'], cwd="..", check=True)
print("All Done")
