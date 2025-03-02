import subprocess

# Define the output file
requirements_file = "requirements_generated.txt"

# Run the pip freeze command and save the output
with open(requirements_file, "w") as f:
    subprocess.run(["pip", "freeze"], stdout=f)

print(f"✅ Requirements file '{requirements_file}' created successfully.")
