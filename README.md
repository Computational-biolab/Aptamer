# "AIoptamer: Artificial Intelligence-Driven Aptamer Optimization Pipeline for Targeted Therapeutics in Healthcare"
This study explores the potential of AI and machine learning in designing aptamers for targeted drug delivery and diagnostics in healthcare. Aptamers are single-stranded oligomers that can specifically bind to targets, playing vital roles in drug discovery, bioimaging, and precision medicine.

**Key Features:**
Objective: Optimize RNA nucleotides for enhanced stability, binding affinity, and therapeutic applications.
Approaches: ML/AI based tools for virtual screeening and binding afinity prediction.
Applications: Cancer treatment, infectious diseases, precision medicine, and targeted therapies.

**We have utlized already existed machine learning and deep learning based tools to prepare a novel pipeline to design new modified aptamers**

**Requirements**
1- Databases: PDB database, AptaDB, UTexas Aptamer Database, RNAapt3D
2- Tools and Software
  (a) Screening Tools: AptaNet, PredPRBA
  (b) Library generation using python script
      #import csv
      #import itertools
  (b) Structure Analysis: UCSF ChimeraX, PyMOL, CHIMERA_NA.
  (c) Simulation Tools: AMBER, GROMACS, Schrodinger for Molecular Dynamics Software
  (d) Feature Prediction: UnaFold, RNAFold, OligoAnalyzer
2- AptaNet script for features extraction and prediction
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
  1- Aptamer library generation : 16384 oligonucleotide sequnces have been generated of 3HSB RNA sequence 'AGAGAGA'.
  2- Virtual screening: AptaNet has been used for virtual screening. 
     (a) Features extraction: Features has been extracted on the basis of interaction features such as AAC (Amino acid composition), k-3mer, PseAAC (Pseudo Amino Acid Composition)  
         will be extracted.
     (b) Prediction: After features extraction prediction code has been used to predict the interaction between the aptamer sequences and proteins. As result the code will give Class 1 (Interaction), Class 0 (No 
         interaction) and interaction probability percentage. On the basis of these predictions the most interaction aptamers can be filtered out using Class 1 and interaction probability percentage.
  3- After filtering 98 oligonucleotide sequences have been left that are showing interaction and have better interaction probability.
  4- These sequences now been mutated using CHIMERA_NA tool, whose commands can be found in the paper.
  5- Use PredPRBA (ML based scoring function) which will take the .pdb format of modified aptamer sequence and protein complex and predict the binding affinity in kcal/mol.

  Note: After the process of complex generation using CHIMERA_NA, the generated complexes were unable to be executed in PredPRBA due to mutations in the aptamer sequence in the initial complex. To overcome this issue, a python script was generated to modify mutated structures and create new complex structures extended with an extension ’ _1’ in PDB format acceptable in PredPRBA. The python script is provided in a supplementary file. After execution of the code, all 98 complexes were fit for execution on PredPRBA for binding affinity prediction. File_processing.py script will be use.
  Reference:
  CHIMERA_NA Tool: Pant, P. (2024). CHIMERA_NA: A Customizable Mutagenesis Tool for Structural Manipulations in Nucleic Acids and Their Complexes. ACS omega, 9(38), 40061-40066.
  PredPRBA link: http://predprba.denglab.org/
  PDA-Pred link: https://web.iitm.ac.in/bioinfo2/pdapred/
