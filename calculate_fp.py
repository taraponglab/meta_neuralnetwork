#importing libraries
import pandas as pd
import os
from padelpy import padeldescriptor
from glob import glob

def convert_to_smi(df, folder_path='tpo',column='canonical_smiles'):
    # Specify the path to your input CSV file
    smiles_column = df[column]
    output_smi_file= os.path.join(folder_path,'smiles.smi')
    smiles_column.to_csv(output_smi_file, index=False, header=False)
    print(f"File converted successfully to {output_smi_file}")
    return output_smi_file

def read_descriptor(folder_path = "fingerprints_xml"):
    xml_files = glob(os.path.join(folder_path, "*.xml"))
    xml_files.sort()
    xml_files
    #set fingerprint list
    FP_list = [
     'AP2DC',
     'AD2D',
     'EState',
     'CDKExt',
     'CDK',
     'CDKGraph',
     'KRFPC',
     'KRFP',
     'MACCS',
     'PubChem',
     'SubFPC',
     'SubFP',]
    fp = dict(zip(FP_list, xml_files))
    fp
    
    return FP_list, fp

def calculate_fp(FP_list, df, output_smi_file, fp, fingerprint_output_dir='fingerprints/test'):
    #Calculate fingerprints
    for i in FP_list:
        fingerprint = i
        fingerprint_output_file = os.path.join(fingerprint_output_dir,''.join([fingerprint,'.csv']))
        fingerprint_descriptortypes = fp[fingerprint]
        padeldescriptor(mol_dir=output_smi_file,
                    d_file=fingerprint_output_file,
                    descriptortypes= fingerprint_descriptortypes,
                    retainorder=True, 
                    removesalt=True,
                    threads=2,
                    detectaromaticity=True,
                    standardizetautomers=True,
                    standardizenitro=True,
                    fingerprints=True
                    )
        Fingerprint = pd.read_csv(fingerprint_output_file)     
        Fingerprint.insert(0, 'LigandID', df['LigandID'])
        Fingerprint = Fingerprint.drop('Name', axis=1)
        Fingerprint.to_csv(fingerprint_output_file, index=False)
        print(fingerprint_output_file, 'done')

def main():
    """
    Calculates fingerprints for chemical compounds based on a given DataFrame.
    ------
    Parameters:
    df      : The filename of the CSV file containing the chemical data.
    name    : The directory where the input file is located and where the output files will be saved.
    ------
    Returns:
    None: Outputs are saved directly to files in the specified directory.
    """
    file_name = input("Type your CSV file name (including extension): ")
    folder_name = input("Type your desired folder path name: ")
    name = folder_name
    if not os.path.exists(name):
        os.makedirs(name)
        print(f"Folder '{name}' created.")
    
    df= pd.read_csv(file_name)
    output_smi_file=convert_to_smi(df, name, column='canonical_smiles')
    FP_list, fp=read_descriptor(folder_path = "fingerprints_xml")
    calculate_fp(FP_list, df, output_smi_file, fp, fingerprint_output_dir=name)
    
    print("Finish calculate 12 fingerprints!")

if __name__=="__main__":
    main()