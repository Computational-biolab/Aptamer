#### AptaNet script for features extraction
import numpy as np
import pandas as pd
from collections import Counter

# Function to calculate k-mer encoding (for k=3 and k=4) of aptamer sequences
def calculate_kmer_encoding(sequence, k=3):
    kmer_dict = {}
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i+k]
        if kmer not in kmer_dict:
            kmer_dict[kmer] = 1
        else:
            kmer_dict[kmer] += 1
    # Normalize k-mer counts by the length of the sequence
    total_kmers = len(sequence) - k + 1
    return {kmer: count / total_kmers for kmer, count in kmer_dict.items()}

# Function to calculate Amino Acid Composition (AAC)
def calculate_aac(sequence):
    aa_count = Counter(sequence)
    total_aa = len(sequence)
    aac = {aa: aa_count.get(aa, 0) / total_aa for aa in "ACDEFGHIKLMNPQRSTVWY"}
    return list(aac.values())

# Function to calculate Pseudo-Amino Acid Composition (PseAAC)
def calculate_pse_aac(sequence, lambda_value=2):
    aa_properties = {
        'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10,
        'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20
    }
    # Basic AAC
    aac = calculate_aac(sequence)
    
    # Sequence order effects (simplified version, only first tier correlations)
    seq_order_effects = []
    for i in range(len(sequence) - lambda_value):
        seq_order_effects.append(np.sum([aa_properties[sequence[i + j]] for j in range(lambda_value)]))
    
    # Normalize sequence order effects
    order_norm = np.mean(seq_order_effects) / len(seq_order_effects) if seq_order_effects else 0
    pse_aac = aac + [order_norm]  # Add sequence order effects to AAC
    
    return pse_aac

# Function to extract features for multiple aptamers from CSV
def extract_features_from_csv(csv_file_path, protein_sequence, output_file):
    aptamer_data = pd.read_csv(csv_file_path)
    features = []
    
    # Iterate over each aptamer sequence in the CSV file
    for seq in aptamer_data['aptamer_sequence']:
        # Calculate k-mer encoding (using 3-mer as an example)
        aptamer_kmer_encoding = calculate_kmer_encoding(seq, k=3)

        # Calculate AAC and PseAAC for the provided protein
        protein_aac = calculate_aac(protein_sequence)
        protein_pse_aac = calculate_pse_aac(protein_sequence)

        # Combine aptamer features and protein features
        combined_features = list(aptamer_kmer_encoding.values()) + protein_aac + protein_pse_aac
        features.append([seq, aptamer_kmer_encoding, list(aptamer_kmer_encoding.values()), protein_aac, protein_pse_aac, combined_features])
    
    # Save to CSV
    features_df = pd.DataFrame(features, columns=[
        'Aptamer Sequence', 'kmer_3_encoding', 'kmer_3_encoding_values', 'protein_aac', 'protein_pse_aac', 'combined_features'
    ])
    features_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")

# Function to extract features for a single aptamer and protein pair
def extract_features_for_single_pair(aptamer_sequence, protein_sequence, output_file):
    # Calculate k-mer encoding for the aptamer (using 3-mer as an example)
    aptamer_kmer_encoding = calculate_kmer_encoding(aptamer_sequence, k=3)

    # Calculate AAC and PseAAC for the protein
    protein_aac = calculate_aac(protein_sequence)
    protein_pse_aac = calculate_pse_aac(protein_sequence)

    # Combine aptamer features and protein features
    combined_features = list(aptamer_kmer_encoding.values()) + protein_aac + protein_pse_aac
    
    # Save to CSV
    features_df = pd.DataFrame([[
        aptamer_sequence, aptamer_kmer_encoding, list(aptamer_kmer_encoding.values()), protein_aac, protein_pse_aac, combined_features
    ]], columns=[
        'Aptamer Sequence', 'kmer_3_encoding', 'kmer_3_encoding_values', 'protein_aac', 'protein_pse_aac', 'combined_features'
    ])
    features_df.to_csv(output_file, index=False)
    print(f"Features saved to {output_file}")

# Main interactive function
def interactive_feature_extraction():
    choice = input("Do you want to extract features for a CSV file or a single aptamer-protein pair? (Enter 'csv' or 'single'): ").strip().lower()
    
    if choice == 'csv':
        csv_file_path = input("Please provide the CSV file path containing the aptamer sequences: ").strip()
        protein_sequence = input("Please provide the target protein sequence: ").strip()
        output_file = input("Please provide the output file path to save features: ").strip()
        
        extract_features_from_csv(csv_file_path, protein_sequence, output_file)
    
    elif choice == 'single':
        aptamer_sequence = input("Please provide the aptamer sequence: ").strip()
        protein_sequence = input("Please provide the target protein sequence: ").strip()
        output_file = input("Please provide the output file path to save features: ").strip()
        
        extract_features_for_single_pair(aptamer_sequence, protein_sequence, output_file)
    
    else:
        print("Invalid choice. Please enter either 'csv' or 'single'.")

# Run the interactive feature extraction
interactive_feature_extraction()

####### AptaNet script for Prediction
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the extracted features (from the uploaded CSV file)
features_file = 'aptamer_feature_result.csv'  # Path to the file where the features were saved
aptamer_data = pd.read_csv(features_file)

# Ensure 'combined_features' column is properly formatted as lists
aptamer_data['combined_features'] = aptamer_data['combined_features'].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Check the lengths of the feature lists
feature_lengths = aptamer_data['combined_features'].apply(len)
print(f"Feature lengths: {feature_lengths.unique()}")

# Find the maximum feature length
max_length = feature_lengths.max()

# Pad or trim lists to ensure they all have the same length
aptamer_data['combined_features'] = aptamer_data['combined_features'].apply(
    lambda x: x + [0] * (max_length - len(x)) if len(x) < max_length else x[:max_length]
)

# Prepare the feature data (X) and labels (y)
X = np.array(aptamer_data['combined_features'].tolist())  # Extract the combined features
# Dummy labels for testing
y = np.random.randint(0, 2, len(aptamer_data))  # Generate random labels for testing purposes

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model using the entire dataset (not just the test set)
mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42)
mlp.fit(X_scaled, y)

# Make predictions on the entire dataset
y_pred = mlp.predict(X_scaled)

# Evaluate the model
accuracy = accuracy_score(y, y_pred)
report = classification_report(y, y_pred)

print("Model Accuracy:", accuracy)
print("Classification Report:\n", report)

# Get predicted probabilities (for ranking)
y_pred_prob = mlp.predict_proba(X_scaled)[:, 1]  # Probabilities for class 1 (interaction)

# Create a dataframe to store the predictions and their corresponding probabilities
predictions_df = pd.DataFrame({
    'Aptamer Sequence': aptamer_data['Aptamer Sequence'],
    'True Label': y,
    'Predicted Probability': y_pred_prob
})

# Rank the interactions by predicted probability
predictions_df['Rank'] = predictions_df['Predicted Probability'].rank(ascending=False, method='dense')

# Sort the dataframe by predicted probability
predictions_df_sorted = predictions_df.sort_values(by='Predicted Probability', ascending=False)

# Save the sorted predictions to a new CSV file
output_file = 'Ranked_Aptamer_Protein_Interactions.csv'  # Path for the new file
predictions_df_sorted.to_csv(output_file, index=False)

# Print confirmation
print(f"Sorted prediction results saved to {output_file}")
