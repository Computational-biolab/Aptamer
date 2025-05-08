import os

def shift_residues_further_right(input_file, output_file):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    corrected_lines = []

    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            # Extract the residue name and chain ID
            residue_name = line[17:20].strip()  # Residue name (e.g., G, A, U, C)
            chain_id = line[21:22]  # Chain ID (e.g., X, F)
            
            # Check if the residue row contains RNA residues (G, A, U, C)
            if residue_name in ['A', 'U', 'G', 'C']:
                # Shift the residue name one more space to the right without moving the chain ID
                line = line[:16] + '   ' + residue_name + line[20:]

        corrected_lines.append(line)

    # Write the corrected lines to the output file
    with open(output_file, 'w') as file:
        file.writelines(corrected_lines)

def process_multiple_pdb_files(folder_path):
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdb"):  # Check if the file is a PDB file
            input_file = os.path.join(folder_path, filename)
            output_file = os.path.join(folder_path, filename.replace('.pdb', '_1.pdb'))  # Append '_1' to the filename
            shift_residues_further_right(input_file, output_file)
            print(f"Processed: {filename} -> {os.path.basename(output_file)}")

# Folder containing PDB files
pdb_folder = "PDBs"  # Replace with the path to your folder containing PDB files

# Run the function for all PDB files in the folder
process_multiple_pdb_files(pdb_folder)
