import pandas as pd
import numpy as np
import os
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Reshape
from keras.layers import Input, Dense, Attention
from joblib import dump
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, roc_auc_score, balanced_accuracy_score, matthews_corrcoef, precision_score
import matplotlib.pyplot as plt

""" 
1. Load x (molecular features) and y (labels)
2. Train baseline models (CNN, BiLSTM, Attention)
3. Meta-learning model (ATT)
4. Evaluate performance of all train and test
5. Y-randomization
6. Applicability domain
7. Permutation importance
""" 

def cnn_model(fingerprint_length):
    model = Sequential()
    model.add(Conv1D(32, kernel_size=3, input_shape=(fingerprint_length,1), activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(64, 3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def bilstm_model(fingerprint_length):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=(fingerprint_length, 1))))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
    return model

def simple_attention(fingerprint_length):
    input_layer = Input(shape=(fingerprint_length,))
    dense_layer = Dense(64, activation='relu')(input_layer)
    reshape_layer = Reshape((1, 64))(dense_layer)                               # Reshape layer to for attention
    attention_layer = Attention(use_scale=True)([reshape_layer, reshape_layer]) # Attention mechanism layer
    attention_output = Reshape((64,))(attention_layer)
    output_layer = Dense(1, activation='sigmoid')(attention_output)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def y_prediction(model, x_train_np, x_train, y_train, col_name):
    y_prob = model.predict(x_train_np)
    y_pred = (y_prob > 0.5).astype(int).reshape(-1)
    print(y_train.shape)
    print(y_pred.shape)
    print(y_pred)
    acc = accuracy_score(y_train, y_pred)
    sen = recall_score(y_train, y_pred)  # Sensitivity is the same as recall
    mcc = matthews_corrcoef(y_train, y_pred)
    f1  = f1_score(y_train, y_pred)
    y_prob = y_prob.reshape(-1, y_prob.shape[-1])
    print(y_prob.shape)
    
    auc = roc_auc_score(y_train, y_prob)
    bcc = balanced_accuracy_score(y_train, y_pred)
    pre = precision_score(y_train, y_pred)
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    spc = tn / (tn + fp)
    
    
    y_prob = pd.DataFrame(y_prob, columns=[col_name]).set_index(x_train.index)
    y_pred_df = pd.DataFrame(y_pred, columns=[col_name]).set_index(x_train.index)

    # Create a DataFrame to store the metrics
    metrics = pd.DataFrame({
        'Accuracy': [acc],
        'Sensitivity': [sen],
        'Specificity': [spc],
        'MCC': [mcc],
        'F1 Score': [f1],
        'AUC': [auc],
        'BACC': [bcc],
        'Precision': [pre]
    }, index=[col_name])
    
    return y_prob, y_pred_df, metrics

def stacked_class(name):

    xat_train = pd.read_csv(os.path.join(name, "train", 'AD2D.csv'     ), index_col=0)
    xes_train = pd.read_csv(os.path.join(name, "train", 'EState.csv'   ), index_col=0)
    xke_train = pd.read_csv(os.path.join(name, "train", 'KRFP.csv'     ), index_col=0)
    xpc_train = pd.read_csv(os.path.join(name, "train", 'PubChem.csv'  ), index_col=0)
    xss_train = pd.read_csv(os.path.join(name, "train", 'SubFP.csv'    ), index_col=0)
    xcd_train = pd.read_csv(os.path.join(name, "train", 'CDKGraph.csv' ), index_col=0)
    xcn_train = pd.read_csv(os.path.join(name, "train", 'CDK.csv'      ), index_col=0)
    xkc_train = pd.read_csv(os.path.join(name, "train", 'KRFPC.csv'    ), index_col=0)
    xce_train = pd.read_csv(os.path.join(name, "train", 'CDKExt.csv'   ), index_col=0)
    xsc_train = pd.read_csv(os.path.join(name, "train", 'SubFPC.csv'   ), index_col=0)
    xac_train = pd.read_csv(os.path.join(name, "train", 'AP2DC.csv'    ), index_col=0)
    xma_train = pd.read_csv(os.path.join(name, "train", 'MACCS.csv'    ), index_col=0)
    y_train   = pd.read_csv(os.path.join(name, "train", "y_train.csv"  ), index_col=0)
    
    xat_test = pd.read_csv(os.path.join( name, "test",  'AD2D.csv'     ), index_col=0)
    xes_test = pd.read_csv(os.path.join( name, "test",  'EState.csv'   ), index_col=0)
    xke_test = pd.read_csv(os.path.join( name, "test",  'KRFP.csv'     ), index_col=0)
    xpc_test = pd.read_csv(os.path.join( name, "test",  'PubChem.csv'  ), index_col=0)
    xss_test = pd.read_csv(os.path.join( name, "test",  'SubFP.csv'    ), index_col=0)
    xcd_test = pd.read_csv(os.path.join( name, "test",  'CDKGraph.csv' ), index_col=0)
    xcn_test = pd.read_csv(os.path.join( name, "test",  'CDK.csv'      ), index_col=0)
    xkc_test = pd.read_csv(os.path.join( name, "test",  'KRFPC.csv'    ), index_col=0)
    xce_test = pd.read_csv(os.path.join( name, "test",  'CDKExt.csv'   ), index_col=0)
    xsc_test = pd.read_csv(os.path.join( name, "test",  'SubFPC.csv'   ), index_col=0)
    xac_test = pd.read_csv(os.path.join( name, "test",  'AP2DC.csv'    ), index_col=0)
    xma_test = pd.read_csv(os.path.join( name, "test",  'MACCS.csv'    ), index_col=0)
    y_test   = pd.read_csv(os.path.join( name, "test",  "y_test.csv"   ), index_col=0)
    
    xat_train_np = np.array(xat_train)
    xes_train_np = np.array(xes_train)
    xke_train_np = np.array(xke_train)
    xpc_train_np = np.array(xpc_train)
    xss_train_np = np.array(xss_train)
    xcd_train_np = np.array(xcd_train)
    xcn_train_np = np.array(xcn_train)
    xkc_train_np = np.array(xkc_train)
    xce_train_np = np.array(xce_train)
    xsc_train_np = np.array(xsc_train)
    xac_train_np = np.array(xac_train)
    xma_train_np = np.array(xma_train)
    y_train = np.array(y_train)
    
    xat_test_np = np.array(xat_test)
    xes_test_np = np.array(xes_test)
    xke_test_np = np.array(xke_test)
    xpc_test_np = np.array(xpc_test)
    xss_test_np = np.array(xss_test)
    xcd_test_np = np.array(xcd_test)
    xcn_test_np = np.array(xcn_test)
    xkc_test_np = np.array(xkc_test)
    xce_test_np = np.array(xce_test)
    xsc_test_np = np.array(xsc_test)
    xac_test_np = np.array(xac_test)
    xma_test_np = np.array(xma_test)
    y_test = np.array(y_test)


    xat_train_split, xat_val_split, yat_train_split, yat_val_split = train_test_split(xat_train_np, y_train, test_size=0.2, random_state=42)
    xes_train_split, xes_val_split, yes_train_split, yes_val_split = train_test_split(xes_train_np, y_train, test_size=0.2, random_state=42)
    xke_train_split, xke_val_split, yke_train_split, yke_val_split = train_test_split(xke_train_np, y_train, test_size=0.2, random_state=42)
    xpc_train_split, xpc_val_split, ypc_train_split, ypc_val_split = train_test_split(xpc_train_np, y_train, test_size=0.2, random_state=42)
    xss_train_split, xss_val_split, yss_train_split, yss_val_split = train_test_split(xss_train_np, y_train, test_size=0.2, random_state=42)
    xcd_train_split, xcd_val_split, ycd_train_split, ycd_val_split = train_test_split(xcd_train_np, y_train, test_size=0.2, random_state=42)
    xcn_train_split, xcn_val_split, ycn_train_split, ycn_val_split = train_test_split(xcn_train_np, y_train, test_size=0.2, random_state=42)
    xkc_train_split, xkc_val_split, ykc_train_split, ykc_val_split = train_test_split(xkc_train_np, y_train, test_size=0.2, random_state=42)
    xce_train_split, xce_val_split, yce_train_split, yce_val_split = train_test_split(xce_train_np, y_train, test_size=0.2, random_state=42)
    xsc_train_split, xsc_val_split, ysc_train_split, ysc_val_split = train_test_split(xsc_train_np, y_train, test_size=0.2, random_state=42)
    xac_train_split, xac_val_split, yac_train_split, yac_val_split = train_test_split(xac_train_np, y_train, test_size=0.2, random_state=42)
    xma_train_split, xma_val_split, yma_train_split, yma_val_split = train_test_split(xma_train_np, y_train, test_size=0.2, random_state=42)

    
    # Train CNN models
    baseline_model_cnn_at = cnn_model(fingerprint_length=xat_train_split.shape[1])
    baseline_model_cnn_es = cnn_model(fingerprint_length=xes_train_split.shape[1])
    baseline_model_cnn_ke = cnn_model(fingerprint_length=xke_train_split.shape[1])
    baseline_model_cnn_pc = cnn_model(fingerprint_length=xpc_train_split.shape[1])
    baseline_model_cnn_ss = cnn_model(fingerprint_length=xss_train_split.shape[1])
    baseline_model_cnn_cd = cnn_model(fingerprint_length=xcd_train_split.shape[1])
    baseline_model_cnn_cn = cnn_model(fingerprint_length=xcn_train_split.shape[1])
    baseline_model_cnn_kc = cnn_model(fingerprint_length=xkc_train_split.shape[1])
    baseline_model_cnn_ce = cnn_model(fingerprint_length=xce_train_split.shape[1])
    baseline_model_cnn_sc = cnn_model(fingerprint_length=xsc_train_split.shape[1])
    baseline_model_cnn_ac = cnn_model(fingerprint_length=xac_train_split.shape[1])
    baseline_model_cnn_ma = cnn_model(fingerprint_length=xma_train_split.shape[1])
    baseline_model_cnn_at.fit(xat_train_split, yat_train_split, validation_data=(xat_val_split, yat_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_es.fit(xes_train_split, yes_train_split, validation_data=(xes_val_split, yes_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_ke.fit(xke_train_split, yke_train_split, validation_data=(xke_val_split, yke_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_pc.fit(xpc_train_split, ypc_train_split, validation_data=(xpc_val_split, ypc_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_ss.fit(xss_train_split, yss_train_split, validation_data=(xss_val_split, yss_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_cd.fit(xcd_train_split, ycd_train_split, validation_data=(xcd_val_split, ycd_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_cn.fit(xcn_train_split, ycn_train_split, validation_data=(xcn_val_split, ycn_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_kc.fit(xkc_train_split, ykc_train_split, validation_data=(xkc_val_split, ykc_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_ce.fit(xce_train_split, yce_train_split, validation_data=(xce_val_split, yce_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_sc.fit(xsc_train_split, ysc_train_split, validation_data=(xsc_val_split, ysc_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_ac.fit(xac_train_split, yac_train_split, validation_data=(xac_val_split, yac_val_split), epochs=20, batch_size=32)
    baseline_model_cnn_ma.fit(xma_train_split, yma_train_split, validation_data=(xma_val_split, yma_val_split), epochs=20, batch_size=32)
    
    # Save the trained models
    baseline_model_cnn_at.save(os.path.join(name, "baseline_model_cnn_at.keras"))
    baseline_model_cnn_es.save(os.path.join(name, "baseline_model_cnn_es.keras"))
    baseline_model_cnn_ke.save(os.path.join(name, "baseline_model_cnn_ke.keras"))
    baseline_model_cnn_pc.save(os.path.join(name, "baseline_model_cnn_pc.keras"))
    baseline_model_cnn_ss.save(os.path.join(name, "baseline_model_cnn_ss.keras"))
    baseline_model_cnn_cd.save(os.path.join(name, "baseline_model_cnn_cd.keras"))
    baseline_model_cnn_cn.save(os.path.join(name, "baseline_model_cnn_cn.keras"))
    baseline_model_cnn_kc.save(os.path.join(name, "baseline_model_cnn_kc.keras"))
    baseline_model_cnn_ce.save(os.path.join(name, "baseline_model_cnn_ce.keras"))
    baseline_model_cnn_sc.save(os.path.join(name, "baseline_model_cnn_sc.keras"))
    baseline_model_cnn_ac.save(os.path.join(name, "baseline_model_cnn_ac.keras"))
    baseline_model_cnn_ma.save(os.path.join(name, "baseline_model_cnn_ma.keras"))
    # Predict with CNN models
    yat_prob_cnn_train, yat_pred_cnn_train, yat_metric_cnn_train = y_prediction(baseline_model_cnn_at, xat_train_np, xat_train, y_train, "yat_pred_cnn")
    yes_prob_cnn_train, yes_pred_cnn_train, yes_metric_cnn_train = y_prediction(baseline_model_cnn_es, xes_train_np, xes_train, y_train, "yes_pred_cnn")
    yke_prob_cnn_train, yke_pred_cnn_train, yke_metric_cnn_train = y_prediction(baseline_model_cnn_ke, xke_train_np, xke_train, y_train, "yke_pred_cnn")
    ypc_prob_cnn_train, ypc_pred_cnn_train, ypc_metric_cnn_train = y_prediction(baseline_model_cnn_pc, xpc_train_np, xpc_train, y_train, "ypc_pred_cnn")
    yss_prob_cnn_train, yss_pred_cnn_train, yss_metric_cnn_train = y_prediction(baseline_model_cnn_ss, xss_train_np, xss_train, y_train, "yss_pred_cnn")
    ycd_prob_cnn_train, ycd_pred_cnn_train, ycd_metric_cnn_train = y_prediction(baseline_model_cnn_cd, xcd_train_np, xcd_train, y_train, "ycd_pred_cnn")
    ycn_prob_cnn_train, ycn_pred_cnn_train, ycn_metric_cnn_train = y_prediction(baseline_model_cnn_cn, xcn_train_np, xcn_train, y_train, "ycn_pred_cnn")
    ykc_prob_cnn_train, ykc_pred_cnn_train, ykc_metric_cnn_train = y_prediction(baseline_model_cnn_kc, xkc_train_np, xkc_train, y_train, "ykc_pred_cnn")
    yce_prob_cnn_train, yce_pred_cnn_train, yce_metric_cnn_train = y_prediction(baseline_model_cnn_ce, xce_train_np, xce_train, y_train, "yce_pred_cnn")
    ysc_prob_cnn_train, ysc_pred_cnn_train, ysc_metric_cnn_train = y_prediction(baseline_model_cnn_sc, xsc_train_np, xsc_train, y_train, "ysc_pred_cnn")
    yac_prob_cnn_train, yac_pred_cnn_train, yac_metric_cnn_train = y_prediction(baseline_model_cnn_ac, xac_train_np, xac_train, y_train, "yac_pred_cnn")
    yma_prob_cnn_train, yma_pred_cnn_train, yma_metric_cnn_train = y_prediction(baseline_model_cnn_ma, xma_train_np, xma_train, y_train, "yma_pred_cnn")
    yat_prob_cnn_test, yat_pred_cnn_test, yat_metric_cnn_test   =  y_prediction(baseline_model_cnn_at, xat_test_np, xat_test, y_test, "yat_pred_cnn")
    yes_prob_cnn_test, yes_pred_cnn_test, yes_metric_cnn_test   =  y_prediction(baseline_model_cnn_es, xes_test_np, xes_test, y_test, "yes_pred_cnn")
    yke_prob_cnn_test, yke_pred_cnn_test, yke_metric_cnn_test   =  y_prediction(baseline_model_cnn_ke, xke_test_np, xke_test, y_test, "yke_pred_cnn")
    ypc_prob_cnn_test, ypc_pred_cnn_test, ypc_metric_cnn_test   =  y_prediction(baseline_model_cnn_pc, xpc_test_np, xpc_test, y_test, "ypc_pred_cnn")
    yss_prob_cnn_test, yss_pred_cnn_test, yss_metric_cnn_test   =  y_prediction(baseline_model_cnn_ss, xss_test_np, xss_test, y_test, "yss_pred_cnn")
    ycd_prob_cnn_test, ycd_pred_cnn_test, ycd_metric_cnn_test   =  y_prediction(baseline_model_cnn_cd, xcd_test_np, xcd_test, y_test, "ycd_pred_cnn")
    ycn_prob_cnn_test, ycn_pred_cnn_test, ycn_metric_cnn_test   =  y_prediction(baseline_model_cnn_cn, xcn_test_np, xcn_test, y_test, "ycn_pred_cnn")
    ykc_prob_cnn_test, ykc_pred_cnn_test, ykc_metric_cnn_test   =  y_prediction(baseline_model_cnn_kc, xkc_test_np, xkc_test, y_test, "ykc_pred_cnn")
    yce_prob_cnn_test, yce_pred_cnn_test, yce_metric_cnn_test   =  y_prediction(baseline_model_cnn_ce, xce_test_np, xce_test, y_test, "yce_pred_cnn")
    ysc_prob_cnn_test, ysc_pred_cnn_test, ysc_metric_cnn_test   =  y_prediction(baseline_model_cnn_sc, xsc_test_np, xsc_test, y_test, "ysc_pred_cnn")
    yac_prob_cnn_test, yac_pred_cnn_test, yac_metric_cnn_test   =  y_prediction(baseline_model_cnn_ac, xac_test_np, xac_test, y_test, "yac_pred_cnn")
    yma_prob_cnn_test, yma_pred_cnn_test, yma_metric_cnn_test   =  y_prediction(baseline_model_cnn_ma, xma_test_np, xma_test, y_test, "yma_pred_cnn")
    
    # Reshaping the training data for BiLSTM
    xat_train_np_bilstm = xat_train_np.reshape((-1, 1, xat_train_np.shape[1]))
    xes_train_np_bilstm = xes_train_np.reshape((-1, 1, xes_train_np.shape[1]))
    xke_train_np_bilstm = xke_train_np.reshape((-1, 1, xke_train_np.shape[1]))
    xpc_train_np_bilstm = xpc_train_np.reshape((-1, 1, xpc_train_np.shape[1]))
    xss_train_np_bilstm = xss_train_np.reshape((-1, 1, xss_train_np.shape[1]))
    xcd_train_np_bilstm = xcd_train_np.reshape((-1, 1, xcd_train_np.shape[1]))
    xcn_train_np_bilstm = xcn_train_np.reshape((-1, 1, xcn_train_np.shape[1]))
    xkc_train_np_bilstm = xkc_train_np.reshape((-1, 1, xkc_train_np.shape[1]))
    xce_train_np_bilstm = xce_train_np.reshape((-1, 1, xce_train_np.shape[1]))
    xsc_train_np_bilstm = xsc_train_np.reshape((-1, 1, xsc_train_np.shape[1]))
    xac_train_np_bilstm = xac_train_np.reshape((-1, 1, xac_train_np.shape[1]))
    xma_train_np_bilstm = xma_train_np.reshape((-1, 1, xma_train_np.shape[1]))

    # Reshaping the test data  for BiLSTM
    xat_test_np_bilstm = xat_test_np.reshape((-1, 1, xat_test_np.shape[1]))
    xes_test_np_bilstm = xes_test_np.reshape((-1, 1, xes_test_np.shape[1]))
    xke_test_np_bilstm = xke_test_np.reshape((-1, 1, xke_test_np.shape[1]))
    xpc_test_np_bilstm = xpc_test_np.reshape((-1, 1, xpc_test_np.shape[1]))
    xss_test_np_bilstm = xss_test_np.reshape((-1, 1, xss_test_np.shape[1]))
    xcd_test_np_bilstm = xcd_test_np.reshape((-1, 1, xcd_test_np.shape[1]))
    xcn_test_np_bilstm = xcn_test_np.reshape((-1, 1, xcn_test_np.shape[1]))
    xkc_test_np_bilstm = xkc_test_np.reshape((-1, 1, xkc_test_np.shape[1]))
    xce_test_np_bilstm = xce_test_np.reshape((-1, 1, xce_test_np.shape[1]))
    xsc_test_np_bilstm = xsc_test_np.reshape((-1, 1, xsc_test_np.shape[1]))
    xac_test_np_bilstm = xac_test_np.reshape((-1, 1, xac_test_np.shape[1]))
    xma_test_np_bilstm = xma_test_np.reshape((-1, 1, xma_test_np.shape[1]))
    
    xat_train_split_bilstm, xat_val_split_bilstm, yat_train_split_bilstm, yat_val_split_bilstm = train_test_split(xat_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xes_train_split_bilstm, xes_val_split_bilstm, yes_train_split_bilstm, yes_val_split_bilstm = train_test_split(xes_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xke_train_split_bilstm, xke_val_split_bilstm, yke_train_split_bilstm, yke_val_split_bilstm = train_test_split(xke_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xpc_train_split_bilstm, xpc_val_split_bilstm, ypc_train_split_bilstm, ypc_val_split_bilstm = train_test_split(xpc_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xss_train_split_bilstm, xss_val_split_bilstm, yss_train_split_bilstm, yss_val_split_bilstm = train_test_split(xss_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xcd_train_split_bilstm, xcd_val_split_bilstm, ycd_train_split_bilstm, ycd_val_split_bilstm = train_test_split(xcd_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xcn_train_split_bilstm, xcn_val_split_bilstm, ycn_train_split_bilstm, ycn_val_split_bilstm = train_test_split(xcn_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xkc_train_split_bilstm, xkc_val_split_bilstm, ykc_train_split_bilstm, ykc_val_split_bilstm = train_test_split(xkc_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xce_train_split_bilstm, xce_val_split_bilstm, yce_train_split_bilstm, yce_val_split_bilstm = train_test_split(xce_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xsc_train_split_bilstm, xsc_val_split_bilstm, ysc_train_split_bilstm, ysc_val_split_bilstm = train_test_split(xsc_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xac_train_split_bilstm, xac_val_split_bilstm, yac_train_split_bilstm, yac_val_split_bilstm = train_test_split(xac_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    xma_train_split_bilstm, xma_val_split_bilstm, yma_train_split_bilstm, yma_val_split_bilstm = train_test_split(xma_train_np_bilstm, y_train, test_size=0.2, random_state=42)
    
    # Train BiLSTM models
    baseline_model_bilstm_at = bilstm_model(fingerprint_length=xat_train.shape[1])
    baseline_model_bilstm_es = bilstm_model(fingerprint_length=xes_train.shape[1])
    baseline_model_bilstm_ke = bilstm_model(fingerprint_length=xke_train.shape[1])
    baseline_model_bilstm_pc = bilstm_model(fingerprint_length=xpc_train.shape[1])
    baseline_model_bilstm_ss = bilstm_model(fingerprint_length=xss_train.shape[1])
    baseline_model_bilstm_cd = bilstm_model(fingerprint_length=xcd_train.shape[1])
    baseline_model_bilstm_cn = bilstm_model(fingerprint_length=xcn_train.shape[1])
    baseline_model_bilstm_kc = bilstm_model(fingerprint_length=xkc_train.shape[1])
    baseline_model_bilstm_ce = bilstm_model(fingerprint_length=xce_train.shape[1])
    baseline_model_bilstm_sc = bilstm_model(fingerprint_length=xsc_train.shape[1])
    baseline_model_bilstm_ac = bilstm_model(fingerprint_length=xac_train.shape[1])
    baseline_model_bilstm_ma = bilstm_model(fingerprint_length=xma_train.shape[1])
    baseline_model_bilstm_at.fit(xat_train_split_bilstm, yat_train_split_bilstm, validation_data=(xat_val_split_bilstm, yat_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_es.fit(xes_train_split_bilstm, yes_train_split_bilstm, validation_data=(xes_val_split_bilstm, yes_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_ke.fit(xke_train_split_bilstm, yke_train_split_bilstm, validation_data=(xke_val_split_bilstm, yke_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_pc.fit(xpc_train_split_bilstm, ypc_train_split_bilstm, validation_data=(xpc_val_split_bilstm, ypc_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_ss.fit(xss_train_split_bilstm, yss_train_split_bilstm, validation_data=(xss_val_split_bilstm, yss_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_cd.fit(xcd_train_split_bilstm, ycd_train_split_bilstm, validation_data=(xcd_val_split_bilstm, ycd_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_cn.fit(xcn_train_split_bilstm, ycn_train_split_bilstm, validation_data=(xcn_val_split_bilstm, ycn_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_kc.fit(xkc_train_split_bilstm, ykc_train_split_bilstm, validation_data=(xkc_val_split_bilstm, ykc_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_ce.fit(xce_train_split_bilstm, yce_train_split_bilstm, validation_data=(xce_val_split_bilstm, yce_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_sc.fit(xsc_train_split_bilstm, ysc_train_split_bilstm, validation_data=(xsc_val_split_bilstm, ysc_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_ac.fit(xac_train_split_bilstm, yac_train_split_bilstm, validation_data=(xac_val_split_bilstm, yac_val_split_bilstm), epochs=20, batch_size=32)
    baseline_model_bilstm_ma.fit(xma_train_split_bilstm, yma_train_split_bilstm, validation_data=(xma_val_split_bilstm, yma_val_split_bilstm), epochs=20, batch_size=32)
    
    # Save the trained models
    baseline_model_bilstm_at.save(os.path.join(name, "baseline_model_bilstm_at.keras"))
    baseline_model_bilstm_es.save(os.path.join(name, "baseline_model_bilstm_es.keras"))
    baseline_model_bilstm_ke.save(os.path.join(name, "baseline_model_bilstm_ke.keras"))
    baseline_model_bilstm_pc.save(os.path.join(name, "baseline_model_bilstm_pc.keras"))
    baseline_model_bilstm_ss.save(os.path.join(name, "baseline_model_bilstm_ss.keras"))
    baseline_model_bilstm_cd.save(os.path.join(name, "baseline_model_bilstm_cd.keras"))
    baseline_model_bilstm_cn.save(os.path.join(name, "baseline_model_bilstm_cn.keras"))
    baseline_model_bilstm_kc.save(os.path.join(name, "baseline_model_bilstm_kc.keras"))
    baseline_model_bilstm_ce.save(os.path.join(name, "baseline_model_bilstm_ce.keras"))
    baseline_model_bilstm_sc.save(os.path.join(name, "baseline_model_bilstm_sc.keras"))
    baseline_model_bilstm_ac.save(os.path.join(name, "baseline_model_bilstm_ac.keras"))
    baseline_model_bilstm_ma.save(os.path.join(name, "baseline_model_bilstm_ma.keras"))
    
    # Predict with BiLSTM models
    yat_prob_bilstm_train, yat_pred_bilstm_train, yat_metric_bilstm_train = y_prediction(   baseline_model_bilstm_at, xat_train_np_bilstm, xat_train, y_train, "yat_pred_bilstm")
    yes_prob_bilstm_train, yes_pred_bilstm_train, yes_metric_bilstm_train = y_prediction(   baseline_model_bilstm_es, xes_train_np_bilstm, xes_train, y_train, "yes_pred_bilstm")
    yke_prob_bilstm_train, yke_pred_bilstm_train, yke_metric_bilstm_train = y_prediction(   baseline_model_bilstm_ke, xke_train_np_bilstm, xke_train, y_train, "yke_pred_bilstm")
    ypc_prob_bilstm_train, ypc_pred_bilstm_train, ypc_metric_bilstm_train = y_prediction(   baseline_model_bilstm_pc, xpc_train_np_bilstm, xpc_train, y_train, "ypc_pred_bilstm")
    yss_prob_bilstm_train, yss_pred_bilstm_train, yss_metric_bilstm_train = y_prediction(   baseline_model_bilstm_ss, xss_train_np_bilstm, xss_train, y_train, "yss_pred_bilstm")
    ycd_prob_bilstm_train, ycd_pred_bilstm_train, ycd_metric_bilstm_train = y_prediction(   baseline_model_bilstm_cd, xcd_train_np_bilstm, xcd_train, y_train, "ycd_pred_bilstm")
    ycn_prob_bilstm_train, ycn_pred_bilstm_train, ycn_metric_bilstm_train = y_prediction(   baseline_model_bilstm_cn, xcn_train_np_bilstm, xcn_train, y_train, "ycn_pred_bilstm")
    ykc_prob_bilstm_train, ykc_pred_bilstm_train, ykc_metric_bilstm_train = y_prediction(   baseline_model_bilstm_kc, xkc_train_np_bilstm, xkc_train, y_train, "ykc_pred_bilstm")
    yce_prob_bilstm_train, yce_pred_bilstm_train, yce_metric_bilstm_train = y_prediction(   baseline_model_bilstm_ce, xce_train_np_bilstm, xce_train, y_train, "yce_pred_bilstm")
    ysc_prob_bilstm_train, ysc_pred_bilstm_train, ysc_metric_bilstm_train = y_prediction(   baseline_model_bilstm_sc, xsc_train_np_bilstm, xsc_train, y_train, "ysc_pred_bilstm")
    yac_prob_bilstm_train, yac_pred_bilstm_train, yac_metric_bilstm_train = y_prediction(   baseline_model_bilstm_ac, xac_train_np_bilstm, xac_train, y_train, "yac_pred_bilstm")
    yma_prob_bilstm_train, yma_pred_bilstm_train, yma_metric_bilstm_train = y_prediction(   baseline_model_bilstm_ma, xma_train_np_bilstm, xma_train, y_train, "yma_pred_bilstm")
    yat_prob_bilstm_test, yat_pred_bilstm_test,  yat_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_at, xat_test_np_bilstm, xat_test, y_test,   "yat_pred_bilstm")
    yes_prob_bilstm_test, yes_pred_bilstm_test,  yes_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_es, xes_test_np_bilstm, xes_test, y_test,   "yes_pred_bilstm")
    yke_prob_bilstm_test, yke_pred_bilstm_test,  yke_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_ke, xke_test_np_bilstm, xke_test, y_test,   "yke_pred_bilstm")
    ypc_prob_bilstm_test, ypc_pred_bilstm_test,  ypc_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_pc, xpc_test_np_bilstm, xpc_test, y_test,   "ypc_pred_bilstm")
    yss_prob_bilstm_test, yss_pred_bilstm_test,  yss_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_ss, xss_test_np_bilstm, xss_test, y_test,   "yss_pred_bilstm")
    ycd_prob_bilstm_test, ycd_pred_bilstm_test,  ycd_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_cd, xcd_test_np_bilstm, xcd_test, y_test,   "ycd_pred_bilstm")
    ycn_prob_bilstm_test, ycn_pred_bilstm_test,  ycn_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_cn, xcn_test_np_bilstm, xcn_test, y_test,   "ycn_pred_bilstm")
    ykc_prob_bilstm_test, ykc_pred_bilstm_test,  ykc_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_kc, xkc_test_np_bilstm, xkc_test, y_test,   "ykc_pred_bilstm")
    yce_prob_bilstm_test, yce_pred_bilstm_test,  yce_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_ce, xce_test_np_bilstm, xce_test, y_test,   "yce_pred_bilstm")
    ysc_prob_bilstm_test, ysc_pred_bilstm_test,  ysc_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_sc, xsc_test_np_bilstm, xsc_test, y_test,   "ysc_pred_bilstm")
    yac_prob_bilstm_test, yac_pred_bilstm_test,  yac_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_ac, xac_test_np_bilstm, xac_test, y_test,   "yac_pred_bilstm")
    yma_prob_bilstm_test, yma_pred_bilstm_test,  yma_metric_bilstm_test   = y_prediction(  baseline_model_bilstm_ma, xma_test_np_bilstm, xma_test, y_test,   "yma_pred_bilstm")
    
    # Train attention models
    baseline_model_att_at = simple_attention(fingerprint_length=xat_train.shape[1])
    baseline_model_att_es = simple_attention(fingerprint_length=xes_train.shape[1])
    baseline_model_att_ke = simple_attention(fingerprint_length=xke_train.shape[1])
    baseline_model_att_pc = simple_attention(fingerprint_length=xpc_train.shape[1])
    baseline_model_att_ss = simple_attention(fingerprint_length=xss_train.shape[1])
    baseline_model_att_cd = simple_attention(fingerprint_length=xcd_train.shape[1])
    baseline_model_att_cn = simple_attention(fingerprint_length=xcn_train.shape[1])
    baseline_model_att_kc = simple_attention(fingerprint_length=xkc_train.shape[1])
    baseline_model_att_ce = simple_attention(fingerprint_length=xce_train.shape[1])
    baseline_model_att_sc = simple_attention(fingerprint_length=xsc_train.shape[1])
    baseline_model_att_ac = simple_attention(fingerprint_length=xac_train.shape[1])
    baseline_model_att_ma = simple_attention(fingerprint_length=xma_train.shape[1])
    baseline_model_att_at.fit(xat_train_split, yat_train_split, validation_data=(xat_val_split, yat_val_split), epochs=20, batch_size=32)
    baseline_model_att_es.fit(xes_train_split, yes_train_split, validation_data=(xes_val_split, yes_val_split), epochs=20, batch_size=32)
    baseline_model_att_ke.fit(xke_train_split, yke_train_split, validation_data=(xke_val_split, yke_val_split), epochs=20, batch_size=32)
    baseline_model_att_pc.fit(xpc_train_split, ypc_train_split, validation_data=(xpc_val_split, ypc_val_split), epochs=20, batch_size=32)
    baseline_model_att_ss.fit(xss_train_split, yss_train_split, validation_data=(xss_val_split, yss_val_split), epochs=20, batch_size=32)
    baseline_model_att_cd.fit(xcd_train_split, ycd_train_split, validation_data=(xcd_val_split, ycd_val_split), epochs=20, batch_size=32)
    baseline_model_att_cn.fit(xcn_train_split, ycn_train_split, validation_data=(xcn_val_split, ycn_val_split), epochs=20, batch_size=32)
    baseline_model_att_kc.fit(xkc_train_split, ykc_train_split, validation_data=(xkc_val_split, ykc_val_split), epochs=20, batch_size=32)
    baseline_model_att_ce.fit(xce_train_split, yce_train_split, validation_data=(xce_val_split, yce_val_split), epochs=20, batch_size=32)
    baseline_model_att_sc.fit(xsc_train_split, ysc_train_split, validation_data=(xsc_val_split, ysc_val_split), epochs=20, batch_size=32)
    baseline_model_att_ac.fit(xac_train_split, yac_train_split, validation_data=(xac_val_split, yac_val_split), epochs=20, batch_size=32)
    baseline_model_att_ma.fit(xma_train_split, yma_train_split, validation_data=(xma_val_split, yma_val_split), epochs=20, batch_size=32)
    
    # Save the trained models
    baseline_model_att_at.save(os.path.join(name, "baseline_model_satt_at.keras"))
    baseline_model_att_es.save(os.path.join(name, "baseline_model_satt_es.keras"))
    baseline_model_att_ke.save(os.path.join(name, "baseline_model_satt_ke.keras"))
    baseline_model_att_pc.save(os.path.join(name, "baseline_model_satt_pc.keras"))
    baseline_model_att_ss.save(os.path.join(name, "baseline_model_satt_ss.keras"))
    baseline_model_att_cd.save(os.path.join(name, "baseline_model_satt_cd.keras"))
    baseline_model_att_cn.save(os.path.join(name, "baseline_model_satt_cn.keras"))
    baseline_model_att_kc.save(os.path.join(name, "baseline_model_satt_kc.keras"))
    baseline_model_att_ce.save(os.path.join(name, "baseline_model_satt_ce.keras"))
    baseline_model_att_sc.save(os.path.join(name, "baseline_model_satt_sc.keras"))
    baseline_model_att_ac.save(os.path.join(name, "baseline_model_satt_ac.keras"))
    baseline_model_att_ma.save(os.path.join(name, "baseline_model_satt_ma.keras"))
    # Predict with attention models
    yat_prob_att_train, yat_pred_att_train, yat_metric_att_train = y_prediction(   baseline_model_att_at, xat_train_np, xat_train, y_train, "yat_pred_att")
    yes_prob_att_train, yes_pred_att_train, yes_metric_att_train = y_prediction(   baseline_model_att_es, xes_train_np, xes_train, y_train, "yes_pred_att")
    yke_prob_att_train, yke_pred_att_train, yke_metric_att_train = y_prediction(   baseline_model_att_ke, xke_train_np, xke_train, y_train, "yke_pred_att")
    ypc_prob_att_train, ypc_pred_att_train, ypc_metric_att_train = y_prediction(   baseline_model_att_pc, xpc_train_np, xpc_train, y_train, "ypc_pred_att")
    yss_prob_att_train, yss_pred_att_train, yss_metric_att_train = y_prediction(   baseline_model_att_ss, xss_train_np, xss_train, y_train, "yss_pred_att")
    ycd_prob_att_train, ycd_pred_att_train, ycd_metric_att_train = y_prediction(   baseline_model_att_cd, xcd_train_np, xcd_train, y_train, "ycd_pred_att")
    ycn_prob_att_train, ycn_pred_att_train, ycn_metric_att_train = y_prediction(   baseline_model_att_cn, xcn_train_np, xcn_train, y_train, "ycn_pred_att")
    ykc_prob_att_train, ykc_pred_att_train, ykc_metric_att_train = y_prediction(   baseline_model_att_kc, xkc_train_np, xkc_train, y_train, "ykc_pred_att")
    yce_prob_att_train, yce_pred_att_train, yce_metric_att_train = y_prediction(   baseline_model_att_ce, xce_train_np, xce_train, y_train, "yce_pred_att")
    ysc_prob_att_train, ysc_pred_att_train, ysc_metric_att_train = y_prediction(   baseline_model_att_sc, xsc_train_np, xsc_train, y_train, "ysc_pred_att")
    yac_prob_att_train, yac_pred_att_train, yac_metric_att_train = y_prediction(   baseline_model_att_ac, xac_train_np, xac_train, y_train, "yac_pred_att")
    yma_prob_att_train, yma_pred_att_train, yma_metric_att_train = y_prediction(   baseline_model_att_ma, xma_train_np, xma_train, y_train, "yma_pred_att")
    yat_prob_att_test, yat_pred_att_test,  yat_metric_att_test   = y_prediction(  baseline_model_att_at, xat_test_np, xat_test, y_test,   "yat_pred_att")
    yes_prob_att_test, yes_pred_att_test,  yes_metric_att_test   = y_prediction(  baseline_model_att_es, xes_test_np, xes_test, y_test,   "yes_pred_att")
    yke_prob_att_test, yke_pred_att_test,  yke_metric_att_test   = y_prediction(  baseline_model_att_ke, xke_test_np, xke_test, y_test,   "yke_pred_att")
    ypc_prob_att_test, ypc_pred_att_test,  ypc_metric_att_test   = y_prediction(  baseline_model_att_pc, xpc_test_np, xpc_test, y_test,   "ypc_pred_att")
    yss_prob_att_test, yss_pred_att_test,  yss_metric_att_test   = y_prediction(  baseline_model_att_ss, xss_test_np, xss_test, y_test,   "yss_pred_att")
    ycd_prob_att_test, ycd_pred_att_test,  ycd_metric_att_test   = y_prediction(  baseline_model_att_cd, xcd_test_np, xcd_test, y_test,   "ycd_pred_att")
    ycn_prob_att_test, ycn_pred_att_test,  ycn_metric_att_test   = y_prediction(  baseline_model_att_cn, xcn_test_np, xcn_test, y_test,   "ycn_pred_att")
    ykc_prob_att_test, ykc_pred_att_test,  ykc_metric_att_test   = y_prediction(  baseline_model_att_kc, xkc_test_np, xkc_test, y_test,   "ykc_pred_att")
    yce_prob_att_test, yce_pred_att_test,  yce_metric_att_test   = y_prediction(  baseline_model_att_ce, xce_test_np, xce_test, y_test,   "yce_pred_att")
    ysc_prob_att_test, ysc_pred_att_test,  ysc_metric_att_test   = y_prediction(  baseline_model_att_sc, xsc_test_np, xsc_test, y_test,   "ysc_pred_att")
    yac_prob_att_test, yac_pred_att_test,  yac_metric_att_test   = y_prediction(  baseline_model_att_ac, xac_test_np, xac_test, y_test,   "yac_pred_att")
    yma_prob_att_test, yma_pred_att_test,  yma_metric_att_test   = y_prediction(  baseline_model_att_ma, xma_test_np, xma_test, y_test,   "yma_pred_att")
    
    # Stack the predictive features
    stack_train_prob_all = pd.concat([yat_prob_cnn_train, yat_prob_bilstm_train, yat_prob_att_train,
                            yes_prob_cnn_train, yes_prob_bilstm_train, yes_prob_att_train,
                            yke_prob_cnn_train, yke_prob_bilstm_train, yke_prob_att_train,
                            ypc_prob_cnn_train, ypc_prob_bilstm_train, ypc_prob_att_train,
                            yss_prob_cnn_train, yss_prob_bilstm_train, yss_prob_att_train,
                            ycd_prob_cnn_train, ycd_prob_bilstm_train, ycd_prob_att_train,
                            ycn_prob_cnn_train, ycn_prob_bilstm_train, ycn_prob_att_train,
                            ykc_prob_cnn_train, ykc_prob_bilstm_train, ykc_prob_att_train,
                            yce_prob_cnn_train, yce_prob_bilstm_train, yce_prob_att_train,
                            ysc_prob_cnn_train, ysc_prob_bilstm_train, ysc_prob_att_train,
                            yac_prob_cnn_train, yac_prob_bilstm_train, yac_prob_att_train,
                            yma_prob_cnn_train, yma_prob_bilstm_train, yma_prob_att_train],  axis=1)
    stack_test_prob_all  = pd.concat([yat_prob_cnn_test, yat_prob_bilstm_test, yat_prob_att_test,
                            yes_prob_cnn_test, yes_prob_bilstm_test, yes_prob_att_test,
                            yke_prob_cnn_test, yke_prob_bilstm_test, yke_prob_att_test,
                            ypc_prob_cnn_test, ypc_prob_bilstm_test, ypc_prob_att_test,
                            yss_prob_cnn_test, yss_prob_bilstm_test, yss_prob_att_test,
                            ycd_prob_cnn_test, ycd_prob_bilstm_test, ycd_prob_att_test,
                            ycn_prob_cnn_test, ycn_prob_bilstm_test, ycn_prob_att_test,
                            ykc_prob_cnn_test, ykc_prob_bilstm_test, ykc_prob_att_test,
                            yce_prob_cnn_test, yce_prob_bilstm_test, yce_prob_att_test,
                            ysc_prob_cnn_test, ysc_prob_bilstm_test, ysc_prob_att_test,
                            yac_prob_cnn_test, yac_prob_bilstm_test, yac_prob_att_test,
                            yma_prob_cnn_test, yma_prob_bilstm_test, yma_prob_att_test],  axis=1)
    stack_test_pred_all  = pd.concat([yat_pred_cnn_test, yat_pred_bilstm_test, yat_pred_att_test,
                        yes_pred_cnn_test, yes_pred_bilstm_test, yes_pred_att_test,
                        yke_pred_cnn_test, yke_pred_bilstm_test, yke_pred_att_test,
                        ypc_pred_cnn_test, ypc_pred_bilstm_test, ypc_pred_att_test,
                        yss_pred_cnn_test, yss_pred_bilstm_test, yss_pred_att_test,
                        ycd_pred_cnn_test, ycd_pred_bilstm_test, ycd_pred_att_test,
                        ycn_pred_cnn_test, ycn_pred_bilstm_test, ycn_pred_att_test,
                        ykc_pred_cnn_test, ykc_pred_bilstm_test, ykc_pred_att_test,
                        yce_pred_cnn_test, yce_pred_bilstm_test, yce_pred_att_test,
                        ysc_pred_cnn_test, ysc_pred_bilstm_test, ysc_pred_att_test,
                        yac_pred_cnn_test, yac_pred_bilstm_test, yac_pred_att_test,
                        yma_pred_cnn_test, yma_pred_bilstm_test, yma_pred_att_test],  axis=1)
    stack_train_pred_all  = pd.concat([yat_pred_cnn_train, yat_pred_bilstm_train, yat_pred_att_train,
                        yes_pred_cnn_train, yes_pred_bilstm_train, yes_pred_att_train,
                        yke_pred_cnn_train, yke_pred_bilstm_train, yke_pred_att_train,
                        ypc_pred_cnn_train, ypc_pred_bilstm_train, ypc_pred_att_train,
                        yss_pred_cnn_train, yss_pred_bilstm_train, yss_pred_att_train,
                        ycd_pred_cnn_train, ycd_pred_bilstm_train, ycd_pred_att_train,
                        ycn_pred_cnn_train, ycn_pred_bilstm_train, ycn_pred_att_train,
                        ykc_pred_cnn_train, ykc_pred_bilstm_train, ykc_pred_att_train,
                        yce_pred_cnn_train, yce_pred_bilstm_train, yce_pred_att_train,
                        ysc_pred_cnn_train, ysc_pred_bilstm_train, ysc_pred_att_train,
                        yac_pred_cnn_train, yac_pred_bilstm_train, yac_pred_att_train,
                        yma_pred_cnn_train, yma_pred_bilstm_train, yma_pred_att_train],  axis=1)
    stack_prob_train = pd.concat ([ypc_prob_att_train, ysc_prob_cnn_train, ysc_prob_bilstm_train],  axis=1)
    stack_prob_test = pd.concat ([ypc_prob_att_test, ysc_prob_cnn_test, ysc_prob_bilstm_test],  axis=1)
    stack_pred_train = pd.concat ([ypc_pred_att_train, ysc_pred_cnn_train, ysc_pred_bilstm_train],  axis=1)
    stack_pred_test = pd.concat ([ypc_pred_att_test, ysc_pred_cnn_test, ysc_pred_bilstm_test],  axis=1)

    stack_train_prob_all.to_csv(os.path.join(name, "all_stacked_train_prob.csv"))
    stack_test_prob_all.to_csv(os.path.join(name, "all_stacked_test_prob.csv"))
    stack_train_pred_all.to_csv(os.path.join(name, "all_stacked_train_predict.csv"))
    stack_test_pred_all.to_csv(os.path.join(name, "all_stacked_test_predict.csv"))

    stack_prob_train.to_csv(os.path.join(name, "stacked_train_prob.csv"))
    stack_prob_test.to_csv(os.path.join(name, "stacked_test_prob.csv"))
    stack_pred_train.to_csv(os.path.join(name, "stacked_train_predict.csv"))
    stack_pred_test.to_csv(os.path.join(name, "stacked_test_predict.csv"))
    
    stack_train_np = np.array(stack_prob_train)
    stack_test_np = np.array(stack_prob_test)
    x_train_stack, x_val_stack, y_train_stack, y_val_stack = train_test_split(stack_train_np, y_train, test_size=0.2, random_state=42)
    stacked_model = simple_attention(fingerprint_length=x_train_stack.shape[1])
    stacked_model.fit(x_train_stack,  y_train_stack, validation_data=(x_val_stack, y_val_stack), epochs=20, batch_size=32)
    stacked_model.save(os.path.join(name, "meta_att_stacked_model.keras"))

    y_prob_stk_train, y_pred_stk_train, y_metric_stk_train = y_prediction(   stacked_model, stack_train_np, stack_prob_train, y_train, "y_pred_stacked")
    y_prob_stk_test, y_pred_stk_test, y_metric_stk_test   = y_prediction(   stacked_model, stack_test_np, stack_prob_test,  y_test,  "y_pred_stacked")
    
    y_prob_stk_train.to_csv(os.path.join( name, "y_prob_train.csv"))
    y_prob_stk_test.to_csv(os.path.join( name, "y_prob_test.csv"))
    y_pred_stk_train.to_csv(os.path.join( name, "y_pred_train.csv"))
    y_pred_stk_test.to_csv(os.path.join( name, "y_pred_test.csv"))

    # Combine performance metrics
    metric_train= pd.concat([yat_metric_cnn_train, yat_metric_bilstm_train, yat_metric_att_train,
                            yes_metric_cnn_train, yes_metric_bilstm_train, yes_metric_att_train,
                            yke_metric_cnn_train, yke_metric_bilstm_train, yke_metric_att_train,
                            ypc_metric_cnn_train, ypc_metric_bilstm_train, ypc_metric_att_train,
                            yss_metric_cnn_train, yss_metric_bilstm_train, yss_metric_att_train,
                            ycd_metric_cnn_train, ycd_metric_bilstm_train, ycd_metric_att_train,
                            ycn_metric_cnn_train, ycn_metric_bilstm_train, ycn_metric_att_train,
                            ykc_metric_cnn_train, ykc_metric_bilstm_train, ykc_metric_att_train,
                            yce_metric_cnn_train, yce_metric_bilstm_train, yce_metric_att_train,
                            ysc_metric_cnn_train, ysc_metric_bilstm_train, ysc_metric_att_train,
                            yac_metric_cnn_train, yac_metric_bilstm_train, yac_metric_att_train,
                            yma_metric_cnn_train, yma_metric_bilstm_train, yma_metric_att_train, y_metric_stk_train],  axis=0)
    metric_test= pd.concat([yat_metric_cnn_test, yat_metric_bilstm_test, yat_metric_att_test,
                            yes_metric_cnn_test, yes_metric_bilstm_test, yes_metric_att_test,
                            yke_metric_cnn_test, yke_metric_bilstm_test, yke_metric_att_test,
                            ypc_metric_cnn_test, ypc_metric_bilstm_test, ypc_metric_att_test,
                            yss_metric_cnn_test, yss_metric_bilstm_test, yss_metric_att_test,
                            ycd_metric_cnn_test, ycd_metric_bilstm_test, ycd_metric_att_test,
                            ycn_metric_cnn_test, ycn_metric_bilstm_test, ycn_metric_att_test,
                            ykc_metric_cnn_test, ykc_metric_bilstm_test, ykc_metric_att_test,
                            yce_metric_cnn_test, yce_metric_bilstm_test, yce_metric_att_test,
                            ysc_metric_cnn_test, ysc_metric_bilstm_test, ysc_metric_att_test,
                            yac_metric_cnn_test, yac_metric_bilstm_test, yac_metric_att_test,
                            yma_metric_cnn_test, yma_metric_bilstm_test, yma_metric_att_test, y_metric_stk_test],  axis=0)
    metric_train.to_csv(os.path.join( name, "metric_train.csv"))
    metric_test.to_csv(os.path.join( name, "metric_test.csv"))

    return stacked_model, stack_prob_train, stack_prob_test, metric_train, metric_test

def nearest_neighbor_AD(x_train, x_test, name, k, z=0.5):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean').fit(x_train)
    dump(nn, os.path.join(name, "ad_"+ str(k) +"_"+ str(z) +".joblib"))
    distance, index = nn.kneighbors(x_train)
    # Calculate mean and sd of distance in train set
    di = np.mean(distance, axis=1)
    # Find mean and sd of di
    dk = np.mean(di)
    sk = np.std(di)
    print('dk = ', dk)
    print('sk = ', sk)
    # Calculate di of test set
    distance, index = nn.kneighbors(x_test)
    di= np.mean(distance, axis=1)
    AD_status = ['within_AD' if di[i] < dk + (z * sk) else 'outside_AD' for i in range(len(di))]
    
    # Create DataFrame with index from x_test and the respective status
    df = pd.DataFrame(AD_status, index=x_test.index, columns=['AD_status'])
    return df, dk, sk

def run_ad(stacked_model, x_train, x_test, stack_test, y_test, name, z = 0.5):
    # Initialize lists to store metrics for plotting
    k_values = [3, 4, 5, 6, 7, 8, 9, 10]    #1, 2, 
    MCC_values = []
    ACC_values = []
    Sen_values = []
    Spe_values = []
    AUC_values = []
    F1_values = []
    BA_values = []
    Prec_values = []
    removed_compounds_values = []
    dk_values = []
    sk_values = []

    # Remove outside AD
    for i in k_values:
        print('k = ', i, 'z=', str(z))
        t, dk, sk = nearest_neighbor_AD(x_train, x_test, name, i, z=z)
        print(t['AD_status'].value_counts())
        # Remove outside AD
        x_ad_test = stack_test[t['AD_status'] == 'within_AD']
        x_ad_test_np = np.array(x_ad_test)
        print(x_ad_test_np.shape)
        y_ad_test = y_test.loc[x_ad_test.index]
        y_prob_test, y_pred_test, y_metric_test = y_prediction(stacked_model, x_ad_test_np, x_ad_test, y_ad_test, name)
        print(len(x_ad_test),len(y_ad_test), len(y_pred_test) )
        # Evaluation
        print('Test set')
        accuracy = round(accuracy_score(y_ad_test, y_pred_test), 3)
        conf_matrix = confusion_matrix(y_ad_test, y_pred_test)
        F1 = round(f1_score(y_ad_test, y_pred_test, average='weighted'), 3)
        tn, fp, fn, tp = conf_matrix.ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        auc = roc_auc_score(y_ad_test, y_prob_test)
        mcc = round(matthews_corrcoef(y_ad_test, y_pred_test), 3)
        balanced_acc = round(balanced_accuracy_score(y_ad_test, y_pred_test), 3)
        prec = precision_score(y_ad_test, y_pred_test)
        print('ACC: ', accuracy, 'Sen: ', sensitivity, 'Spe: ', specificity, 'AUC: ', auc, 'MCC: ', mcc, 'F1: ', F1, 'BA: ', balanced_acc, 'Precision', prec)
        # Store metrics for plotting
        MCC_values.append(mcc)
        ACC_values.append(accuracy)
        Sen_values.append(sensitivity)
        Spe_values.append(specificity)
        AUC_values.append(auc)
        F1_values.append(F1)
        BA_values.append(balanced_acc)
        Prec_values.append(prec)
        removed_compounds_values.append((t['AD_status'] == 'outside_AD').sum())
        dk_values.append(dk)
        sk_values.append(sk)
    k_values   = np.array(k_values)
    MCC_values = np.array(MCC_values)
    ACC_values = np.array(ACC_values)
    Sen_values = np.array(Sen_values)
    Spe_values = np.array(Spe_values)
    AUC_values = np.array(AUC_values)
    F1_values  = np.array(F1_values)
    BA_values  = np.array(BA_values)
    Prec_values = np.array(Prec_values)
    removed_compounds_values = np.array(removed_compounds_values)
    dk_values = np.array(dk_values)
    sk_values = np.array(sk_values)
    # Save table
    ad_metrics = pd.DataFrame({
        "k": k_values[:len(MCC_values)],  # Adjust if some values are skipped
        "Accuracy": ACC_values,
        "Balanced Accuracy": BA_values,
        "Sensitivity": Sen_values,
        "Specificity": Spe_values,
        "MCC": MCC_values,
        "AUC": AUC_values,
        "Precision": Prec_values,
        "F1 Score": F1_values,
        "Removed Compounds": removed_compounds_values,
        "dk_values": dk_values,
        "sk_values": sk_values
    })
    ad_metrics = round(ad_metrics, 3)
    ad_metrics.to_csv("AD_metrics_"+name+"_"+ str(z)+ ".csv")
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    ax1.plot(k_values, MCC_values, 'r^-', label = "MCC")
    ax1.plot(k_values, Sen_values, 'gs-', label = "Sen")
    ax1.plot(k_values, Spe_values, 'y*-', label = "Spe")
    ax1.plot(k_values, AUC_values, 'md-', label = "AUC")
    ax1.plot(k_values, BA_values, 'bo-',  label = "BA")
    ax1.plot(k_values, Prec_values, linestyle='-', marker='<', color='orange', label = "Prec")
    # Adding labels and title
    ax1.set_xlabel('k',      fontsize=12, fontstyle='italic',weight="bold")
    ax1.set_ylabel('Scores', fontsize=12, fontstyle='italic', weight='bold')
    ax1.set_xticks(k_values)
    ax1.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1.02))
    # Second plot: Bar plot for removed_compounds_values
    ax2.bar(k_values, removed_compounds_values, color='green', edgecolor='black', alpha=0.5, width=0.3)
    ax2.set_xlabel('k', fontsize=12, fontstyle='italic',weight="bold")
    ax2.set_ylabel('Removed compounds', fontsize=12, fontstyle='italic', weight='bold')
    ax2.set_xticks(k_values)
    plt.tight_layout()
    save_path = os.path.join(name, "AD_TPO_"+ str(z)+ "_Classification_separated.svg")
    plt.savefig(save_path, bbox_inches='tight') 
    plt.close

def y_random(stack_train, stack_test, y_train, y_test, metric_train, metric_test, name):
    '''
    This function is to run the y_randomization
    '''
    MCC_test=[]
    MCC_train=[]
    for i in range(1,101):
      y_train=y_train.sample(frac=1,replace=False,random_state=0)
      y_test=y_test.sample(frac=1,replace=False,random_state=0)
      stack_train_np = np.array(stack_train)
      stack_test_np = np.array(stack_test)
      # Chosen meta-ATT model
      model = simple_attention(fingerprint_length=stack_train.shape[1])
      y_prob_stk_train, y_pred_stk_train, y_metric_stk_test = y_prediction(model, stack_train_np, stack_train, y_train, "y_pred_stacked")
      y_prob_stk_test, y_pred_stk_test, y_metric_stk_test = y_prediction(model, stack_test_np, stack_test, y_test,  "y_pred_stacked")
      MCCext=matthews_corrcoef(y_test, y_pred_stk_test)
      MCC_test.append(MCCext)
      MCCtrain=matthews_corrcoef(y_train, y_pred_stk_train)
      MCC_train.append(MCCtrain)
    size=[50]
    sizes=[20]
    x=[metric_train.loc['y_pred_stacked', 'MCC']]
    y=[metric_test.loc['y_pred_stacked', 'MCC']]
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.axvline(0.5, c='black', ls=':')
    ax.axhline(0.5, c='black', ls=':')
    ax.scatter(x,y,s=size,c=['red'],marker='x', label='Our model')
    ax.scatter(MCC_train,MCC_test, c='blue',edgecolors='black', alpha=0.7, s=sizes, label='Y-randomization')
    ax.set_xlabel('$MCC_{Train}$', fontsize=14,  fontstyle='italic', weight='bold')
    ax.set_ylabel('$MCC_{Test}$', fontsize=14,  fontstyle='italic', weight='bold')
    ax.legend(loc='lower right',fontsize='small')
    # Adjust layout
    plt.tight_layout()
    save_path = os.path.join(name, "Y-randomization-3pfs-classification.svg")
    plt.savefig(save_path, bbox_inches='tight')
    # Show the plots
    plt.close()

def permutation_importance(model, x_test, y_test, n_repeats=10):
    def get_predictions(X):
        y_pred = model.predict(X)
        return np.argmax(y_pred, axis=1) if y_pred.ndim > 1 and y_pred.shape[1] > 1 else (y_pred > 0.5).astype(int)
    
    baseline_score = accuracy_score(y_test, get_predictions(x_test))
    importances = np.zeros(x_test.shape[1])
    
    for i in range(x_test.shape[1]):
        score_diffs = []
        for _ in range(n_repeats):
            x_test_perm = x_test.copy()
            np.random.shuffle(x_test_perm[:, i])
            permuted_score = accuracy_score(y_test, get_predictions(x_test_perm))
            score_diffs.append(baseline_score - permuted_score)
        importances[i] = np.mean(score_diffs)
    
    return importances

def train_permutation(name, x_train, x_test, y_train, y_test, feature_names,epochs=50, batch_size=2):
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train).ravel()
    y_test = np.array(y_test).ravel()
    # Chosen meta-ATT model
    model = simple_attention(fingerprint_length=x_train.shape[1])
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.3)
    model_filename = os.path.join(name, 'stack_permutation.keras')
    model.save(model_filename)
    
    importance_scores = permutation_importance(model, x_test, y_test)
    sorted_indices = np.argsort(importance_scores)[::-1]
    sorted_features = [feature_names[i] for i in sorted_indices]
    sorted_importance_scores = [importance_scores[i] for i in sorted_indices]
    
    return importance_scores, sorted_features, sorted_importance_scores

def run_permutation(name, x_train, x_test, y_train, y_test):
    feature_names = x_test.columns.tolist()
    importance_scores, sorted_features, sorted_importance_scores = train_permutation(name, x_train, x_test, y_train, y_test, feature_names)

    df_sorted_features = pd.DataFrame({
        'Feature': sorted_features,
        'Importance Score': sorted_importance_scores
    })
    save_features = os.path.join(name, 'stack_importance.csv')
    df_sorted_features.to_csv(save_features, index=False)
    print('Sorted features with importance scores saved successfully!')

    print("Feature Names:", feature_names)
    print("Importance Scores:", importance_scores)
    print("Sorted Features:", sorted_features)
    print("Sorted Importance Scores:", sorted_importance_scores)

    top_n = 10
    print(f"Top {top_n} Feature Importance Ranking:")
    for i in range(min(top_n, len(sorted_features))):
        feature = sorted_features[i]
        importance_score = sorted_importance_scores[i]
        print(f"{i+1}. {feature}: {importance_score:.3f}")
        
    print("Feature Importance Scores:")
    for i, score in enumerate(importance_scores):
        print(f"Feature {i}: Score {score:.3f}")
    
    df_sorted_features = df_sorted_features.sort_values(by='Importance Score', ascending=False)
    top_features = df_sorted_features.head(10)
    print(top_features)
    plt.figure(figsize=(4, 4))
    plt.barh(top_features['Feature'], top_features['Importance Score'], color='skyblue')
    plt.xlabel('Permutation Importance', fontsize=12, fontstyle='italic',weight="bold")
    plt.ylabel('Features'              , fontsize=12, fontstyle='italic',weight="bold")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    save_fig = os.path.join(name, 'stack_permutation.svg')
    plt.savefig(save_fig, format='svg')
    # Show the plots
    plt.close()

def main():
    for name in ['TPO_training_data']:
        print("#"*100) 
        print(name)
        y_train  = pd.read_csv(os.path.join(name, "train", "y_train.csv"), index_col=0)
        y_test   = pd.read_csv(os.path.join(name, "test", "y_test.csv" ), index_col=0)
        print(y_train)
        stacked_model, stack_train, stack_test, metric_train, metric_test = stacked_class(name)
        print("Finish training model ", name)
        stack_train = pd.read_csv(os.path.join(name, "stacked_train_prob.csv"), index_col=0)
        stack_test  = pd.read_csv(os.path.join(name, "stacked_test_prob.csv"), index_col=0)
        print (stack_test.shape)
        metric_train = pd.read_csv(os.path.join( name, "metric_train.csv"))
        metric_test = pd.read_csv(os.path.join( name, "metric_test.csv"))
        metric_train = metric_train.set_index('Unnamed: 0')
        metric_test = metric_test.set_index('Unnamed: 0')
        y_random(stack_train, stack_test, y_train, y_test, metric_train, metric_test, name)
        print("Finish y-randomization ", name)
        stacked_model =  load_model(os.path.join(name, "meta_att_stacked_model.keras"))
        z_values = [3.5, 4.0]
        x_train = pd.read_csv(os.path.join(name, "train",  'SubFPC.csv'   ), index_col=0)
        x_test  = pd.read_csv(os.path.join( name, "test",  'SubFPC.csv'   ), index_col=0)
        for z in z_values:
            run_ad(stacked_model, x_train, x_test, stack_test, y_test, name, z = z)
        print("Finish applicability domain ", name)
        run_permutation(name, stack_train, stack_test, y_train, y_test)
        print("Finish permutation importance ", name)

if __name__ == "__main__":
    main() 