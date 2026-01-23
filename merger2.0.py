import os
import datetime

# --- Configuration ---
# You can easily change these settings if needed.

# 1. Name of the final output file.
OUTPUT_FILENAME = "project_code_dump.txt"

# 2. List of file extensions to include in the dump.
#    (Only text-based files that are part of your source code)
EXTENSIONS_TO_INCLUDE = [
    '.py', '.csv', '.txt', '.cfg', '.env', '.md'
]

# 3. List of FOLDERS to completely ignore.
#    Updated to match your project's specific folder structure.
FOLDERS_TO_EXCLUDE = [
    '__pycache__',
    'Lib',          # <-- THE REAL FIX
    'Scripts',      # <-- THE REAL FIX
    'Include',      # <-- THE REAL FIX
    '.git',         # Good to keep for version control
    'receipts',     # From your original config
]

# 4. List of specific FILES to ignore.
FILES_TO_EXCLUDE = [
    OUTPUT_FILENAME,     # Don't include the previous dump file
    'merger2.0.py'       # Don't include this script itself.
]
# --- End of Configuration ---


def create_project_dump():
    """
    Walks through the current directory and creates a single text file
    containing the content of all relevant source code files.
    """
    project_root = os.getcwd()
    print(f"Starting project dump from folder: {project_root}")
    
    files_added_count = 0

    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as dump_file:
        
        header = f"""======================================================================
=                  PROJECT SOURCE CODE DUMP                          =
======================================================================

Project Folder: {project_root}
Date Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

This file contains a concatenation of all relevant source files
found in the project directory and its subdirectories.

======================================================================
"""
        dump_file.write(header)

        for root, dirs, files in os.walk(project_root, topdown=True):
            
            # This line uses the corrected FOLDERS_TO_EXCLUDE list
            dirs[:] = [d for d in dirs if d not in FOLDERS_TO_EXCLUDE]
            
            for filename in sorted(files):
                if (filename.endswith(tuple(EXTENSIONS_TO_INCLUDE)) and
                        filename not in FILES_TO_EXCLUDE):
                    
                    file_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(file_path, project_root)
                    
                    print(f"  [+] Adding: {relative_path}")
                    files_added_count += 1
                    
                    dump_file.write("\n\n")
                    dump_file.write(f"<------------------------------>\n")
                    dump_file.write(f"FILE PATH: {relative_path}\n")
                    dump_file.write(f"<------------------------------>\n\n")
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as source_file:
                            dump_file.write(source_file.read())
                    except Exception as e:
                        dump_file.write(f"[Error reading file: {e}]")

    print("\n======================================================================")
    if files_added_count > 0:
        print(f"\nAll done! Added {files_added_count} file(s).")
        print(f"Project dumped successfully into: {OUTPUT_FILENAME}")
    else:
        print("\nWarning: No files were found to add.")
        print("Please check your EXTENSIONS_TO_INCLUDE and FOLDERS_TO_EXCLUDE settings.")
    print("\n======================================================================")


if __name__ == "__main__":
    create_project_dump()