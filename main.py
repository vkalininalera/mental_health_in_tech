import subprocess

# List of scripts to run in order
scripts = [
    "fixing_missing_values.py",
    "data_preparation_encoding.py",
    "Kmeans.py",
    "summary_interpret.py"
]

# Run each script sequentially
for script in scripts:
    print(f"\n--- Running {script} ---")
    try:
        subprocess.run(["python", script], check=True)
        print(f"✅ Finished {script}")
    except subprocess.CalledProcessError:
        print(f"❌ Error occurred in {script}, stopping pipeline.")
        break