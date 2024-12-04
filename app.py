import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Title and Introduction
st.title("Predicting Solubility of Molecules")
st.markdown("""
In this app, we predict the solubility of molecules using Linear Regression. 
This is a vital step in **Drug Discovery and Development**, as solubility plays a key role in drug design.
""")

# Upload dataset
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your dataset (CSV format)", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:")
    st.write(data.head())

    # Convert SMILES to RDKit objects
    st.write("Converting SMILES strings to RDKit objects...")
    data['RDKit_Molecule'] = data['SMILES'].apply(Chem.MolFromSmiles)

    # Calculate molecular descriptors
    def generate_descriptors(smiles):
        moldata = [Chem.MolFromSmiles(sm) for sm in smiles]
        descriptors = []
        for mol in moldata:
            desc_MolLogP = Descriptors.MolLogP(mol)
            desc_MolWt = Descriptors.MolWt(mol)
            desc_NumRotatableBonds = Descriptors.NumRotatableBonds(mol)
            descriptors.append([desc_MolLogP, desc_MolWt, desc_NumRotatableBonds])
        return pd.DataFrame(descriptors, columns=["MolLogP", "MolWt", "NumRotatableBonds"])

    st.write("Calculating Descriptors:")
    descriptor_df = generate_descriptors(data['SMILES'])
    st.write(descriptor_df.head())

    def calculate_aromatic_proportion(m):
        aromatic_atoms = [m.GetAtomWithIdx(i).GetIsAromatic() for i in range(m.GetNumAtoms())]
        aa_count = sum(aromatic_atoms)
        return aa_count / Descriptors.HeavyAtomCount(m) if Descriptors.HeavyAtomCount(m) > 0 else 0

    data['AromaticProportion'] = data['RDKit_Molecule'].apply(calculate_aromatic_proportion)
    st.write("Aromatic Proportion Added:")
    st.write(data[['SMILES', 'AromaticProportion']].head())

    # Combine descriptors into X
    X = pd.concat([descriptor_df, data['AromaticProportion']], axis=1)
    st.write("Final Descriptor Data:")
    st.write(X.head())

    # Define Y
    Y = data.iloc[:, 1]  # Assuming the second column is the target variable
    st.write("Target Variable (Y):")
    st.write(Y.head())

    # Split data
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    st.write("Data Split:")
    st.write(f"Training Data: {X_train.shape[0]} samples")
    st.write(f"Testing Data: {X_test.shape[0]} samples")

    # Train Linear Regression model
    model = linear_model.LinearRegression()
    model.fit(X_train, Y_train)

    st.write("Model Trained:")
    st.write("Coefficients:", model.coef_)
    st.write("Intercept:", model.intercept_)

    # Predict and evaluate
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)
    st.write("Evaluation on Training Data:")
    st.write(f"Mean Squared Error: {mean_squared_error(Y_train, Y_pred_train):.2f}")
    st.write(f"R-squared: {r2_score(Y_train, Y_pred_train):.2f}")

    st.write("Evaluation on Testing Data:")
    st.write(f"Mean Squared Error: {mean_squared_error(Y_test, Y_pred_test):.2f}")
    st.write(f"R-squared: {r2_score(Y_test, Y_pred_test):.2f}")

    # Visualize results
    st.write("Visualizations:")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot for training data
    axes[0].scatter(Y_train, Y_pred_train, c="#7CAE00", alpha=0.3)
    z = np.polyfit(Y_train, Y_pred_train, 1)
    p = np.poly1d(z)
    axes[0].plot(Y_train, p(Y_train), "#F8766D")
    axes[0].set_title("Training Data")
    axes[0].set_xlabel("Experimental LogS")
    axes[0].set_ylabel("Predicted LogS")

    # Plot for testing data
    axes[1].scatter(Y_test, Y_pred_test, c="#619CFF", alpha=0.3)
    z = np.polyfit(Y_test, Y_pred_test, 1)
    p = np.poly1d(z)
    axes[1].plot(Y_test, p(Y_test), "#F8766D")
    axes[1].set_title("Testing Data")
    axes[1].set_xlabel("Experimental LogS")
    axes[1].set_ylabel("Predicted LogS")

    st.pyplot(fig)
else:
    st.warning("Please upload a dataset to proceed.")
