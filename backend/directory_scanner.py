import os

# --- Configuration ---

# The name of the output markdown file
OUTPUT_MD_FILE = "project_snapshot.md"

# File extensions to include (as a tuple)
FILES_TO_INCLUDE = ('.py', '.toml')

# Specific filenames to include
REQUIREMENTS_FILE = 'requirements.txt'

# Directories to ignore completely
DIRS_TO_IGNORE = {
    '.git', 
    '__pycache__', 
    '.venv', 
    'node_modules', 
    '.idea', 
    '.vscode'
}

# ---------------------

def get_lang(filename):
    """Determine the markdown language for a given filename."""
    if filename.endswith('.py'):
        return 'python'
    if filename.endswith('.toml'):
        return 'toml'
    if filename == REQUIREMENTS_FILE:
        return 'text'
    # Default to 'text' if unknown
    return 'text'

def main():
    """Main function to generate the project snapshot."""
    
    start_path = '.'
    tree_structure = []
    files_to_read = []
    
    print("Scanning project structure...")

    # os.walk traverses the directory tree top-down
    for root, dirs, files in os.walk(start_path, topdown=True):
        
        # 1. Filter out ignored directories
        # We modify 'dirs' in-place to prevent os.walk from descending into them
        dirs[:] = [d for d in dirs if d not in DIRS_TO_IGNORE]
        
        # 2. Calculate the indentation level for the tree
        level = root.replace(start_path, '').count(os.sep)
        
        # 3. Add the current directory to the tree structure
        if root == start_path:
            tree_structure.append(".\n")
        else:
            # Add directory to tree with indentation
            indent = '    ' * (level - 1)
            tree_structure.append(f"{indent}└── {os.path.basename(root)}/\n")
            
        file_indent = '    ' * level
        
        # 4. Process files in the current directory
        sorted_files = sorted(files)
        
        for f in sorted_files:
            # Check if this file is one we want to read
            if f.endswith(FILES_TO_INCLUDE) or f == REQUIREMENTS_FILE:
                # Add its path to the list of files to read
                files_to_read.append(os.path.join(root, f))
                # Add the file to the tree structure
                tree_structure.append(f"{file_indent}    ├── {f}\n")
            
            # Optional: You can uncomment the 'elif' block below to also
            # list files that are being skipped in the tree.
            
            # elif f != OUTPUT_MD_FILE and not f.endswith('.lock'):
            #     # Show skipped files for a more complete tree
            #     tree_structure.append(f"{file_indent}    ├── {f} (skipped)\n")

    # 5. Read the contents of all collected files
    print(f"Found {len(files_to_read)} files to include.")
    file_contents = []
    for filepath in files_to_read:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get the language for syntax highlighting
            lang = get_lang(os.path.basename(filepath))
            
            # Normalize the path for consistent display (e.g., './src/main.py')
            display_path = os.path.normpath(filepath)
            
            file_contents.append((display_path, content, lang))
        except Exception as e:
            print(f"Warning: Could not read file {filepath}: {e}")
            
    # 6. Write everything to the output Markdown file
    print(f"Writing snapshot to {OUTPUT_MD_FILE}...")
    try:
        with open(OUTPUT_MD_FILE, 'w', encoding='utf-8') as md_file:
            md_file.write("# Project Snapshot\n\n")
            
            # Write Directory Structure
            md_file.write("## Directory Structure\n\n")
            md_file.write("```text\n")
            md_file.writelines(tree_structure)
            md_file.write("```\n\n")
            
            # Write File Contents
            md_file.write("## File Contents\n\n")
            
            # Sort files by path for a consistent order in the MD file
            for path, content, lang in sorted(file_contents, key=lambda x: x[0]):
                md_file.write(f"### {path}\n\n")
                md_file.write(f"```{lang}\n")
                md_file.write(content)
                md_file.write(f"\n```\n\n")
                
        print(f"Successfully generated project snapshot: {OUTPUT_MD_FILE}")
        
    except Exception as e:
        print(f"Error: Could not write to markdown file: {e}")

if __name__ == "__main__":
    main()
