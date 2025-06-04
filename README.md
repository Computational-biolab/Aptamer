# "AIoptamer: Artificial Intelligence-Driven Aptamer Optimization Pipeline for Targeted Therapeutics in Healthcare"
This study explores the potential of AI and machine learning in designing and optimizing aptamers for targeted drug delivery and diagnostics in healthcare. Aptamers are single-stranded oligomers that can specifically bind to targets, playing vital roles in drug discovery, biosensors, bioimaging, and precision medicine.

**Key Features:**
Objective: Optimize RNA and DNA oligonucleotides for enhanced stability, binding affinity, and therapeutic applications.
Approaches: ML/AI-based tools for virtual screening and binding affinity prediction.
Applications: Cancer treatment, infectious diseases, precision medicine, and targeted therapies.

**We utilized already existing machine learning and deep learning based tools to prepare a novel pipeline to design new modified aptamers.**

**Requirements**
1- Databases: PDB database, AptaDB, UTexas Aptamer Database, RNAapt3D
2- Tools and Software
  (a) Screening Tools: AptaNet, PredPRBA
  (b) Library generation using Python script
      #import csv
      #import itertools
  (b) Structure Analysis: UCSF ChimeraX, PyMOL, CHIMERA_NA.
  (c) Simulation Tools: AMBER, GROMACS, Schrodinger for Molecular Dynamics Software
  (d) Feature Prediction: UnaFold, RNAFold, OligoAnalyzer
2- AptaNet script for feature extraction and prediction
  (a) Languages: Python (for script generation and sequence analysis)
  (b) Libraries: NumPy, SciPy, Matplotlib, Pandas, repDNA
         #sklearn.model_selection import train_test_split
         #sklearn.preprocessing import StandardScaler
         #sklearn.neural_network import MLPClassifier
         #sklearn.metrics import accuracy_score, classification_report
3- System Requirements
  (a) Processor: Minimum quad-core processor
  (b) Memory: At least 8 GB RAM
  (c) Storage: 10 GB free disk space
  (d) Operating System: Windows 10, macOS, or Linux

 ** Case Study results**
  1- Aptamer library generation: 16384 oligonucleotide sequences have been generated for the 3HSB RNA sequence 'AGAGAGA'.
  2- Virtual screening: AptaNet has been used for virtual screening. 
     (a) Feature extraction: Features have been extracted based on interaction features such as AAC (Amino acid composition), k-3mer, PseAAC (Pseudo Amino Acid Composition) will be extracted.
     (b) Prediction: After feature extraction prediction code is used to predict the interaction between the aptamer sequences and proteins. As result, the code will give Class 1 (Interaction), Class 0 (No interaction) and interaction probability percentage. Based on these predictions, the most interacting aptamers can be filtered out using Class 1 and interaction probability percentage.
  3- After filtering, 98 oligonucleotide sequences were selectted showing positive interaction with higher interaction probabilities.
  4- The initial RNA sequence was then mutated using the CHIMERA_NA tool, to all selected 98 sequences generating total 98 target-aptamer complexes. Command files for the CHIMERA_NA tool can be found in their reference paper. 
  5- ML based scoring functions PredPRBA and PDA-Pred were used for estimating binding affinity of the target with the for RNA and DNA apatmers respectively in kcal/mol.
  6- Top 10 complexes were subjected for Molecular Dynamics simulation for analysing energetics.
(Note: After the process of complex generation using CHIMERA_NA, the generated complexes were unable to be executed in PredPRBA due to mutations in the aptamer sequence in the initial complex. To overcome this issue, a Python script was generated to modify mutated structures and create new complex structures extended with an extension ’ _1’ in PDB format acceptable in PredPRBA. The Python script is provided in a supplementary file. After execution of the code, all 98 complexes were fit for execution on PredPRBA for binding affinity prediction. The File_processing.py script was used.)
  Reference:
  CHIMERA_NA Tool: Pant, P. (2024). CHIMERA_NA: A Customizable Mutagenesis Tool for Structural Manipulations in Nucleic Acids and Their Complexes. ACS omega, 9(38), 40061-40066.
  PredPRBA link: http://predprba.denglab.org/
  PDA-Pred link: https://web.iitm.ac.in/bioinfo2/pdapred/
