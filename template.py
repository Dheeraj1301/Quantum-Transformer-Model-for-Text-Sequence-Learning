import os

# Define your project name
project_name = "quantum-error-corrector"

# Define all file paths to be created
list_of_files = [
    f"{project_name}/simulation/generate_circuits.py",
    f"{project_name}/simulation/simulate_fidelity.py",
    
    f"{project_name}/datasets/tokenizer.py",
    f"{project_name}/datasets/data_loader.py",
    
    f"{project_name}/model/transformer_model.py",
    
    f"{project_name}/training/loss_functions.py",
    f"{project_name}/training/train.py",
    
    f"{project_name}/evaluation/evaluate.py",
    
    f"{project_name}/integration/cirq_optimizer.py",

    f"{project_name}/checkpoints/.gitkeep",

    f"{project_name}/requirements.txt",
    f"{project_name}/main.py",
    f"{project_name}/README.md",
    f"{project_name}/.gitignore",
]

# Loop through each file path
for filepath in list_of_files:
    filepath = os.path.normpath(filepath)
    filedir, filename = os.path.split(filepath)

    # Create the directory if it doesn't exist
    if filedir and not os.path.exists(filedir):
        os.makedirs(filedir, exist_ok=True)
        print(f"✅ Created directory: {filedir}")

    # Create the file if it doesn't exist or is empty
    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            pass  # Creates an empty file
        print(f"📄 Created empty file: {filepath}")
    else:
        print(f"⚠️ File already exists and is not empty: {filepath}")
