import pandas as pd
import numpy as np
import os
import calculate_fp as cf
import preprocess as pp
#from rdkit.Chem import AllChem as Chem
from tensorflow.keras.models import load_model

"""
1. Input SMILES
2. Data preprocess
3. Calculate fingeprints
4. Load model
5. Predict new dataset
6. Print out AD
"""

def preprocess(df, name):
    """
    Preprocess a DataFrame by remove missing data, convert SMILES to canonical SMILES,
    and remove inorganic, mixtrues, and duplicates.
    ------
    Parameters:
    df      : DataFrame containing chemical data, including a 'SMILES' column.
    name    : The name of the directory where the preprocessed DataFrame will be saved.
    ------
    Returns:
    df_preprocess   : The filename of the preprocessed CSV file ('output_preprocess.csv').
    None            : If the 'LigandID' column is missing.
    """
    df_selection = pp.remove_missing_data(df,'SMILES')
    df_selection = pp.canonical_smiles(df_selection,'SMILES')
    df_selection = pp.remove_inorganic(df_selection,'canonical_smiles')
    df_selection = pp.remove_mixtures(df_selection,'canonical_smiles')
    df_selection = pp.process_duplicate(df_selection,'canonical_smiles',remove_duplicate=True)
    # Ensure LigandID is preserved
    if 'LigandID' not in df_selection.columns:
        print("Error: 'LigandID' column is missing in the DataFrame.")
        return
    df_preprocess='output_preprocess.csv'
    df_selection.to_csv(os.path.join(name, df_preprocess))
    return df_preprocess

def calculate_fingerprint(df, name):
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
    df= pd.read_csv(os.path.join(name, df))
    output_smi_file=cf.convert_to_smi(df, name, column='canonical_smiles')
    FP_list, fp=cf.read_descriptor(folder_path = "fingerprints_xml")
    cf.calculate_fp(FP_list, df, output_smi_file, fp, fingerprint_output_dir=name)
    
def y_prediction(model, x_train_np, x_train, col_name):
    """
    Generates predictions and probabilities for a given model and input data.
    ------
    Parameters:
    model       : The trained model used for making predictions.
    x_train_np  : Input data in NumPy array format for the model.
    x_train     : Input data in DataFrame format, used to preserve indexing.
    col_name    : Column name for the output DataFrame representing predictions or probabilities.
    ------
    Returns:
    y_prob_df   : DataFrame containing the predicted probabilities.
    y_pred_df   : DataFrame containing the binary predictions.
    """
    y_prob = model.predict(x_train_np)
    y_pred = (y_prob > 0.5).astype(int).reshape(-1) 
    y_prob = y_prob.reshape(-1, y_prob.shape[-1])
    y_prob_df = pd.DataFrame(y_prob, columns=[col_name]).set_index(x_train.index)
    y_pred_df = pd.DataFrame(y_pred, columns=[col_name]).set_index(x_train.index)
    
    return y_prob_df, y_pred_df

def load(name):
    """
    Running prediction with provided stack ensemble learning model.
    ------
    Parameters:
    name    : The directory where the input file is located and where the output files will be saved.
    """
    xat_data = pd.read_csv(os.path.join( name,  'AD2D.csv'     ), index_col=0)
    xes_data = pd.read_csv(os.path.join( name,  'EState.csv'   ), index_col=0)
    xke_data = pd.read_csv(os.path.join( name,  'KRFP.csv'     ), index_col=0)
    xpc_data = pd.read_csv(os.path.join( name,  'PubChem.csv'  ), index_col=0)
    xss_data = pd.read_csv(os.path.join( name,  'SubFP.csv'    ), index_col=0)
    xcd_data = pd.read_csv(os.path.join( name,  'CDKGraph.csv' ), index_col=0)
    xcn_data = pd.read_csv(os.path.join( name,  'CDK.csv'      ), index_col=0)
    xkc_data = pd.read_csv(os.path.join( name,  'KRFPC.csv'    ), index_col=0)
    xce_data = pd.read_csv(os.path.join( name,  'CDKExt.csv'   ), index_col=0)
    xsc_data = pd.read_csv(os.path.join( name,  'SubFPC.csv'   ), index_col=0)
    xac_data = pd.read_csv(os.path.join( name,  'AP2DC.csv'    ), index_col=0)
    xma_data = pd.read_csv(os.path.join( name,  'MACCS.csv'    ), index_col=0)

    xat_data_np = np.array(xat_data)
    xes_data_np = np.array(xes_data)
    xke_data_np = np.array(xke_data)
    xpc_data_np = np.array(xpc_data)
    xss_data_np = np.array(xss_data)
    xcd_data_np = np.array(xcd_data)
    xcn_data_np = np.array(xcn_data)
    xkc_data_np = np.array(xkc_data)
    xce_data_np = np.array(xce_data)
    xsc_data_np = np.array(xsc_data)
    xac_data_np = np.array(xac_data)
    xma_data_np = np.array(xma_data)
    
    # CNN architecture
    baseline_model_cnn_at = load_model(os.path.join("model", "baseline_model_cnn_at.keras"))
    baseline_model_cnn_es = load_model(os.path.join("model", "baseline_model_cnn_es.keras"))
    baseline_model_cnn_ke = load_model(os.path.join("model", "baseline_model_cnn_ke.keras"))
    baseline_model_cnn_pc = load_model(os.path.join("model", "baseline_model_cnn_pc.keras"))
    baseline_model_cnn_ss = load_model(os.path.join("model", "baseline_model_cnn_ss.keras"))
    baseline_model_cnn_cd = load_model(os.path.join("model", "baseline_model_cnn_cd.keras"))
    baseline_model_cnn_cn = load_model(os.path.join("model", "baseline_model_cnn_cn.keras"))
    baseline_model_cnn_kc = load_model(os.path.join("model", "baseline_model_cnn_kc.keras"))
    baseline_model_cnn_ce = load_model(os.path.join("model", "baseline_model_cnn_ce.keras"))
    baseline_model_cnn_sc = load_model(os.path.join("model", "baseline_model_cnn_sc.keras"))
    baseline_model_cnn_ac = load_model(os.path.join("model", "baseline_model_cnn_ac.keras"))
    baseline_model_cnn_ma = load_model(os.path.join("model", "baseline_model_cnn_ma.keras"))

    yat_prob_cnn, yat_pred_cnn   =  y_prediction(baseline_model_cnn_at, xat_data_np, xat_data, "yat_pred_cnn")
    yes_prob_cnn, yes_pred_cnn   =  y_prediction(baseline_model_cnn_es, xes_data_np, xes_data, "yes_pred_cnn")
    yke_prob_cnn, yke_pred_cnn   =  y_prediction(baseline_model_cnn_ke, xke_data_np, xke_data, "yke_pred_cnn")
    ypc_prob_cnn, ypc_pred_cnn   =  y_prediction(baseline_model_cnn_pc, xpc_data_np, xpc_data, "ypc_pred_cnn")
    yss_prob_cnn, yss_pred_cnn   =  y_prediction(baseline_model_cnn_ss, xss_data_np, xss_data, "yss_pred_cnn")
    ycd_prob_cnn, ycd_pred_cnn   =  y_prediction(baseline_model_cnn_cd, xcd_data_np, xcd_data, "ycd_pred_cnn")
    ycn_prob_cnn, ycn_pred_cnn   =  y_prediction(baseline_model_cnn_cn, xcn_data_np, xcn_data, "ycn_pred_cnn")
    ykc_prob_cnn, ykc_pred_cnn   =  y_prediction(baseline_model_cnn_kc, xkc_data_np, xkc_data, "ykc_pred_cnn")
    yce_prob_cnn, yce_pred_cnn   =  y_prediction(baseline_model_cnn_ce, xce_data_np, xce_data, "yce_pred_cnn")
    ysc_prob_cnn, ysc_pred_cnn   =  y_prediction(baseline_model_cnn_sc, xsc_data_np, xsc_data, "ysc_pred_cnn")
    yac_prob_cnn, yac_pred_cnn   =  y_prediction(baseline_model_cnn_ac, xac_data_np, xac_data, "yac_pred_cnn")
    yma_prob_cnn, yma_pred_cnn   =  y_prediction(baseline_model_cnn_ma, xma_data_np, xma_data, "yma_pred_cnn")
    
    # BiLSTM architecture
    xat_data_np_bilstm = xat_data_np.reshape((-1, 1, xat_data_np.shape[1]))
    xes_data_np_bilstm = xes_data_np.reshape((-1, 1, xes_data_np.shape[1]))
    xke_data_np_bilstm = xke_data_np.reshape((-1, 1, xke_data_np.shape[1]))
    xpc_data_np_bilstm = xpc_data_np.reshape((-1, 1, xpc_data_np.shape[1]))
    xss_data_np_bilstm = xss_data_np.reshape((-1, 1, xss_data_np.shape[1]))
    xcd_data_np_bilstm = xcd_data_np.reshape((-1, 1, xcd_data_np.shape[1]))
    xcn_data_np_bilstm = xcn_data_np.reshape((-1, 1, xcn_data_np.shape[1]))
    xkc_data_np_bilstm = xkc_data_np.reshape((-1, 1, xkc_data_np.shape[1]))
    xce_data_np_bilstm = xce_data_np.reshape((-1, 1, xce_data_np.shape[1]))
    xsc_data_np_bilstm = xsc_data_np.reshape((-1, 1, xsc_data_np.shape[1]))
    xac_data_np_bilstm = xac_data_np.reshape((-1, 1, xac_data_np.shape[1]))
    xma_data_np_bilstm = xma_data_np.reshape((-1, 1, xma_data_np.shape[1]))
    
    baseline_model_bilstm_at = load_model(os.path.join("model", "baseline_model_bilstm_at.keras"))
    baseline_model_bilstm_es = load_model(os.path.join("model", "baseline_model_bilstm_es.keras"))
    baseline_model_bilstm_ke = load_model(os.path.join("model", "baseline_model_bilstm_ke.keras"))
    baseline_model_bilstm_pc = load_model(os.path.join("model", "baseline_model_bilstm_pc.keras"))
    baseline_model_bilstm_ss = load_model(os.path.join("model", "baseline_model_bilstm_ss.keras"))
    baseline_model_bilstm_cd = load_model(os.path.join("model", "baseline_model_bilstm_cd.keras"))
    baseline_model_bilstm_cn = load_model(os.path.join("model", "baseline_model_bilstm_cn.keras"))
    baseline_model_bilstm_kc = load_model(os.path.join("model", "baseline_model_bilstm_kc.keras"))
    baseline_model_bilstm_ce = load_model(os.path.join("model", "baseline_model_bilstm_ce.keras"))
    baseline_model_bilstm_sc = load_model(os.path.join("model", "baseline_model_bilstm_sc.keras"))
    baseline_model_bilstm_ac = load_model(os.path.join("model", "baseline_model_bilstm_ac.keras"))
    baseline_model_bilstm_ma = load_model(os.path.join("model", "baseline_model_bilstm_ma.keras"))
    
    yat_prob_bilstm, yat_pred_bilstm   = y_prediction(  baseline_model_bilstm_at, xat_data_np_bilstm, xat_data,   "yat_pred_bilstm")
    yes_prob_bilstm, yes_pred_bilstm   = y_prediction(  baseline_model_bilstm_es, xes_data_np_bilstm, xes_data,   "yes_pred_bilstm")
    yke_prob_bilstm, yke_pred_bilstm   = y_prediction(  baseline_model_bilstm_ke, xke_data_np_bilstm, xke_data,   "yke_pred_bilstm")
    ypc_prob_bilstm, ypc_pred_bilstm   = y_prediction(  baseline_model_bilstm_pc, xpc_data_np_bilstm, xpc_data,   "ypc_pred_bilstm")
    yss_prob_bilstm, yss_pred_bilstm   = y_prediction(  baseline_model_bilstm_ss, xss_data_np_bilstm, xss_data,   "yss_pred_bilstm")
    ycd_prob_bilstm, ycd_pred_bilstm   = y_prediction(  baseline_model_bilstm_cd, xcd_data_np_bilstm, xcd_data,   "ycd_pred_bilstm")
    ycn_prob_bilstm, ycn_pred_bilstm   = y_prediction(  baseline_model_bilstm_cn, xcn_data_np_bilstm, xcn_data,   "ycn_pred_bilstm")
    ykc_prob_bilstm, ykc_pred_bilstm   = y_prediction(  baseline_model_bilstm_kc, xkc_data_np_bilstm, xkc_data,   "ykc_pred_bilstm")
    yce_prob_bilstm, yce_pred_bilstm   = y_prediction(  baseline_model_bilstm_ce, xce_data_np_bilstm, xce_data,   "yce_pred_bilstm")
    ysc_prob_bilstm, ysc_pred_bilstm   = y_prediction(  baseline_model_bilstm_sc, xsc_data_np_bilstm, xsc_data,   "ysc_pred_bilstm")
    yac_prob_bilstm, yac_pred_bilstm   = y_prediction(  baseline_model_bilstm_ac, xac_data_np_bilstm, xac_data,   "yac_pred_bilstm")
    yma_prob_bilstm, yma_pred_bilstm   = y_prediction(  baseline_model_bilstm_ma, xma_data_np_bilstm, xma_data,   "yma_pred_bilstm")
    
    # Attention architecture
    baseline_model_att_at = load_model(os.path.join("model", "baseline_model_att_at.keras"))
    baseline_model_att_es = load_model(os.path.join("model", "baseline_model_att_es.keras"))
    baseline_model_att_ke = load_model(os.path.join("model", "baseline_model_att_ke.keras"))
    baseline_model_att_pc = load_model(os.path.join("model", "baseline_model_att_pc.keras"))
    baseline_model_att_ss = load_model(os.path.join("model", "baseline_model_att_ss.keras"))
    baseline_model_att_cd = load_model(os.path.join("model", "baseline_model_att_cd.keras"))
    baseline_model_att_cn = load_model(os.path.join("model", "baseline_model_att_cn.keras"))
    baseline_model_att_kc = load_model(os.path.join("model", "baseline_model_att_kc.keras"))
    baseline_model_att_ce = load_model(os.path.join("model", "baseline_model_att_ce.keras"))
    baseline_model_att_sc = load_model(os.path.join("model", "baseline_model_att_sc.keras"))
    baseline_model_att_ac = load_model(os.path.join("model", "baseline_model_att_ac.keras"))
    baseline_model_att_ma = load_model(os.path.join("model", "baseline_model_att_ma.keras"))
    
    yat_prob_att, yat_pred_att   = y_prediction(  baseline_model_att_at, xat_data_np, xat_data,   "yat_pred_att")
    yes_prob_att, yes_pred_att   = y_prediction(  baseline_model_att_es, xes_data_np, xes_data,   "yes_pred_att")
    yke_prob_att, yke_pred_att   = y_prediction(  baseline_model_att_ke, xke_data_np, xke_data,   "yke_pred_att")
    ypc_prob_att, ypc_pred_att   = y_prediction(  baseline_model_att_pc, xpc_data_np, xpc_data,   "ypc_pred_att")
    yss_prob_att, yss_pred_att   = y_prediction(  baseline_model_att_ss, xss_data_np, xss_data,   "yss_pred_att")
    ycd_prob_att, ycd_pred_att   = y_prediction(  baseline_model_att_cd, xcd_data_np, xcd_data,   "ycd_pred_att")
    ycn_prob_att, ycn_pred_att   = y_prediction(  baseline_model_att_cn, xcn_data_np, xcn_data,   "ycn_pred_att")
    ykc_prob_att, ykc_pred_att   = y_prediction(  baseline_model_att_kc, xkc_data_np, xkc_data,   "ykc_pred_att")
    yce_prob_att, yce_pred_att   = y_prediction(  baseline_model_att_ce, xce_data_np, xce_data,   "yce_pred_att")
    ysc_prob_att, ysc_pred_att   = y_prediction(  baseline_model_att_sc, xsc_data_np, xsc_data,   "ysc_pred_att")
    yac_prob_att, yac_pred_att   = y_prediction(  baseline_model_att_ac, xac_data_np, xac_data,   "yac_pred_att")
    yma_prob_att, yma_pred_att   = y_prediction(  baseline_model_att_ma, xma_data_np, xma_data,   "yma_pred_att")
    
    # Save predictive features
    stack_data_prob_all = pd.concat([yat_prob_cnn, yat_prob_bilstm, yat_prob_att,
                            yes_prob_cnn, yes_prob_bilstm, yes_prob_att,
                            yke_prob_cnn, yke_prob_bilstm, yke_prob_att,
                            ypc_prob_cnn, ypc_prob_bilstm, ypc_prob_att,
                            yss_prob_cnn, yss_prob_bilstm, yss_prob_att,
                            ycd_prob_cnn, ycd_prob_bilstm, ycd_prob_att,
                            ycn_prob_cnn, ycn_prob_bilstm, ycn_prob_att,
                            ykc_prob_cnn, ykc_prob_bilstm, ykc_prob_att,
                            yce_prob_cnn, yce_prob_bilstm, yce_prob_att,
                            ysc_prob_cnn, ysc_prob_bilstm, ysc_prob_att,
                            yac_prob_cnn, yac_prob_bilstm, yac_prob_att,
                            yma_prob_cnn, yma_prob_bilstm, yma_prob_att],  axis=1)
    stack_data_pred_all  = pd.concat([yat_pred_cnn, yat_pred_bilstm, yat_pred_att,
                        yes_pred_cnn, yes_pred_bilstm, yes_pred_att,
                        yke_pred_cnn, yke_pred_bilstm, yke_pred_att,
                        ypc_pred_cnn, ypc_pred_bilstm, ypc_pred_att,
                        yss_pred_cnn, yss_pred_bilstm, yss_pred_att,
                        ycd_pred_cnn, ycd_pred_bilstm, ycd_pred_att,
                        ycn_pred_cnn, ycn_pred_bilstm, ycn_pred_att,
                        ykc_pred_cnn, ykc_pred_bilstm, ykc_pred_att,
                        yce_pred_cnn, yce_pred_bilstm, yce_pred_att,
                        ysc_pred_cnn, ysc_pred_bilstm, ysc_pred_att,
                        yac_pred_cnn, yac_pred_bilstm, yac_pred_att,
                        yma_pred_cnn, yma_pred_bilstm, yma_pred_att],  axis=1)
    stack_data_prob = pd.concat ([ypc_prob_att, ysc_prob_cnn, ysc_prob_bilstm],  axis=1)
    stack_data_pred = pd.concat ([ypc_pred_att, ysc_pred_cnn, ysc_pred_bilstm],  axis=1)
    
    stack_data_prob_all.to_csv(os.path.join(name, "all_stacked_data_prob.csv"))
    stack_data_pred_all.to_csv(os.path.join(name, "all_stacked_data_predict.csv"))
    stack_data_prob.to_csv(os.path.join(name, "stacked_data_prob.csv"))
    stack_data_pred.to_csv(os.path.join(name, "stacked_data_predict.csv"))
    
    # Meta-model architecture
    stack_data_prob = pd.read_csv(os.path.join(name, "stacked_data_prob.csv"), index_col=0)
    stack_data_prob_np = np.array(stack_data_prob)
    meta_model = load_model(os.path.join("model", "meta_att_stacked_model.keras"))
    y_prob_stk, y_pred_stk   = y_prediction(   meta_model, stack_data_prob_np, stack_data_prob,  "y_pred_stacked")
    y_prob_stk.to_csv(os.path.join( name, "y_prob_stack.csv"))
    y_pred_stk.to_csv(os.path.join( name, "y_pred_stack.csv"))

def print_AD(x_data, z=2.5):
    """
    Evaluates the Applicability Domain (AD) of a dataset using a pre-trained nearest neighbor model.

    Parameters:
    - name (str): The directory where the pre-trained nearest neighbor model (`ad_2_1.0.joblib`) is stored.
    - x_data (DataFrame): The input data for which the AD status needs to be determined.
    - z (float): The z-factor for scaling the applicability domain boundary.
    
    Returns:
    - df (DataFrame): A DataFrame with one column, 'AD_status', indicating whether each sample is within or outside the applicability domain.
    """
    from joblib import load
    nn = load(os.path.join("model", "ad_7_2.5.joblib"))
    distance, index = nn.kneighbors(x_data)
    di = np.mean(distance, axis=1)
    dk =  4.497417768993063
    sk =  2.953309991188386
    print('dk = ', dk)
    print('sk = ', sk)
    AD_status = ['within_AD' if di[i] < dk + (z * sk) else 'outside_AD' for i in range(len(di))]

    # Create DataFrame with index from x_test and the respective status
    df = pd.DataFrame(AD_status, index=x_data.index, columns=['AD_status'])
    print(df['AD_status'].value_counts())
    return df

def evaluate_AD(stacked_model,  stack_data, x_data, name, z=2.5):
    t = print_AD(x_data, z=z)
    print(t['AD_status'].value_counts())
    
    # Remove outside AD
    x_ad_test = stack_data[t['AD_status'] == 'within_AD']
    x_ad_test_np = np.array(x_ad_test)
    y_prob_test = stacked_model.predict(x_ad_test_np)
    y_pred_test = (y_prob_test >= 0.5).astype(int)
    y_pred_df = pd.DataFrame(y_pred_test, columns=["y_pred_AD"]).set_index(x_ad_test.index)
    y_pred_df.to_csv(os.path.join( name, "y_pred_AD.csv"))  
    print(len(x_ad_test), len(y_pred_test))

def main():
    # The input csv file should contain LigandID and SMILES columns
    input_file = input("Type your CSV file name (including extension): ")    # Based on input dataset

    folder_name = input("Type your desired folder path name: ") # Folder_name based on dataset
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")

    # Load the DataFrame from the CSV file
    df = pd.read_csv(input_file)
    for name in [folder_name]:
        print("#"*100)
        print(name)
        df_preprocess=preprocess(df, name)
        calculate_fingerprint(df_preprocess, name)
        print("Finish calculate 12 fingerprints!")
        load(name)
        print("Finish predict all!")
        print("Processing AD....")
        stacked_model = load_model(os.path.join("model", "meta_att_stacked_model.keras"))
        stack_data_prob = pd.read_csv(os.path.join(name, "stacked_data_prob.csv"), index_col=0)
        x_data = pd.read_csv(os.path.join(name, "SubFPC.csv"), index_col=0)
        evaluate_AD(stacked_model, stack_data_prob, x_data, name, z=2.5)
        print("Finish AD !")
        print("Finish TPO_predict !")

if __name__ == "__main__":
    main()