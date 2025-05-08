import itertools
import os

def get_possible_bases(seq_type):
    return ['A', 'U', 'G', 'C'] if seq_type.upper() == "RNA" else ['A', 'T', 'G', 'C']

def generate_mutations(sequence, positions, seq_type):
    sequence = sequence.upper()
    bases = get_possible_bases(seq_type)

    combinations = list(itertools.product(bases, repeat=len(positions)))
    mutated_seqs = []

    for combo in combinations:
        seq_list = list(sequence)
        for i, pos in enumerate(positions):
            seq_list[pos - 1] = combo[i]
        mutated_seqs.append("".join(seq_list))
    
    return mutated_seqs

def main():
    seq_type = input("Enter the type of aptamer sequence (DNA/RNA) (case insensitive): ").strip().upper()
    sequence = input("Enter the nucleotide sequence (e.g. acgggcag) (case insensitive): ").strip().upper()
    
    pos_input = input("Enter residue number to mutate (comma-separated, 1-based): ").strip()
    positions = [int(x) for x in pos_input.split(',')]

    mutated_seqs = generate_mutations(sequence, positions, seq_type)

    print(f"\nGenerated {len(mutated_seqs)} mutated sequences:")
    for i, mut_seq in enumerate(mutated_seqs, 1):
        print(f"{i}: {mut_seq}")

    output_file = "mutated_sequences.csv"
    with open(output_file, "w") as f:
        f.write("aptamer_sequence\n")  # Header line
        for mut_seq in mutated_seqs:
            f.write(mut_seq + "\n")

    abs_path = os.path.abspath(output_file)
    print(f"\n✅ All mutated sequences saved to '{output_file}'.")
    print(f"📂 Full file path: {abs_path}")

if __name__ == "__main__":
    main()
