import os

base_dir = r"D:\Rahul\Git\ML-Projects"

structure = {
    "linear-regression": ["notebooks", "data", "src"],
}

# Create base dir
os.makedirs(base_dir, exist_ok=True)

# Create subfolders and READMEs
for project, subdirs in structure.items():
    project_path = os.path.join(base_dir, project)
    os.makedirs(project_path, exist_ok=True)
    
    # Add subdirs
    for sub in subdirs:
        os.makedirs(os.path.join(project_path, sub), exist_ok=True)
    
    # Project-level README
    with open(os.path.join(project_path, "README.md"), "w") as f:
        f.write(f"# {project.replace('-', ' ').title()}\n\nDescription here.\n")

# Root README
with open(os.path.join(base_dir, "README.md"), "w") as f:
    f.write("# ML Projects\n\nThis repo contains my ML/AI study projects.\n")
