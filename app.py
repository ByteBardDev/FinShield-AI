# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import shap

st.set_page_config(layout='wide')

@st.cache_resource
def load_artifacts(artifact_dir='artifacts'):
    model = joblib.load(f"{artifact_dir}/model.joblib")
    with open(f"{artifact_dir}/feature_columns.json", 'r') as f:
        feature_cols = json.load(f)
    return model, feature_cols


model, feature_columns = load_artifacts()

st.title("Credit Card Fraud Detection")
st.write("Upload CSV with same features (columns) as training data. Include `Class` if you want metrics to be shown.")

uploaded = st.file_uploader("Upload CSV", type=['csv'])
threshold = st.sidebar.slider("Decision threshold", 0.0, 1.0, 0.5)
show_shap_checkbox = st.sidebar.checkbox("Enable SHAP explanations (may be slow)", value=False)

if uploaded is not None:
    df = pd.read_csv(uploaded)

    # Ensure columns exist
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        st.error(f"Uploaded CSV is missing feature columns: {missing}")
    else:
        X = df[feature_columns]
        # Predict probabilities
        try:
            probs = model.predict_proba(X)[:, 1]
        except Exception as e:
            st.error("Model prediction failed. Make sure uploaded features match training features exactly and are in the same order.")
            st.exception(e)
            st.stop()

        preds = (probs >= threshold).astype(int)
        df_result = df.copy()
        df_result['fraud_prob'] = probs
        df_result['predicted_fraud'] = preds

        st.markdown("### Prediction samples")
        st.dataframe(df_result.sort_values('fraud_prob', ascending=False).head(20))

        st.markdown("### Summary")
        st.write("Predicted fraud count:", int(df_result['predicted_fraud'].sum()))
        st.write("Total rows:", len(df_result))

        if 'Class' in df_result.columns:
            cm = confusion_matrix(df_result['Class'], df_result['predicted_fraud'])
            st.write("Confusion matrix (rows: true, cols: pred)")
            st.write(cm)

            # quick metrics
            from sklearn.metrics import precision_score, recall_score, f1_score
            st.write("Precision:", precision_score(df_result['Class'], df_result['predicted_fraud'], zero_division=0))
            st.write("Recall:", recall_score(df_result['Class'], df_result['predicted_fraud'], zero_division=0))
            st.write("F1:", f1_score(df_result['Class'], df_result['predicted_fraud'], zero_division=0))

        # SHAP explanation for top N rows (expensive)
        if show_shap_checkbox:
            with st.spinner('Computing SHAP...'):
                # Prepare the transform part: use the pipeline steps 'imputer' and 'scaler' to transform inputs
                # The model is an imblearn pipeline: imputer -> scaler -> smote -> clf
                # We will manually apply imputer+scaler to get the same input to the clf
                # But since the pipeline contains those objects, we can access them directly
                try:
                    imputer = model.named_steps['imputer']
                    scaler = model.named_steps['scaler']
                    clf = model.named_steps['clf']

                    # transform X to the input space expected by the classifier
                    X_trans = imputer.transform(X)
                    X_trans = scaler.transform(X_trans)

                    explainer = shap.TreeExplainer(clf)

                    # Use a small sample for performance
                    sample_size = min(500, X_trans.shape[0])
                    sample_idx = np.random.choice(X_trans.shape[0], size=sample_size, replace=False)
                    X_sample_trans = X_trans[sample_idx]
                    X_sample_df = X.iloc[sample_idx]

                    # compute shap values (for tree models this is fastish)
                    shap_values = explainer.shap_values(X_sample_trans)

                    # summary plot
                    st.subheader("SHAP summary (global feature importance)")
                    plt.figure(figsize=(8, 5))
                    shap.summary_plot(shap_values, X_sample_df, show=False)
                    st.pyplot(plt)
                    plt.clf()

                    # allow local explanation on a selected row from uploaded CSV
                    st.subheader("Local explanation (select a transaction index)")
                    idx_options = list(range(len(df_result)))
                    selected_idx = st.number_input("Row index to explain", min_value=0, max_value=len(df_result)-1, value=idx_options[0], step=1)
                    if st.button("Show local SHAP for selected row"):
                        # transform single row
                        single_trans = imputer.transform(X.iloc[[selected_idx]])
                        single_trans = scaler.transform(single_trans)
                        single_shap = explainer.shap_values(single_trans)

                        st.write("Transaction (features):")
                        st.write(X.iloc[[selected_idx]].T)

                        st.write("Model probability of fraud:", float(df_result.loc[selected_idx, 'fraud_prob']))
                        plt.figure(figsize=(8,4))
                        shap.force_plot(explainer.expected_value, single_shap, X.iloc[[selected_idx]], matplotlib=True, show=False)
                        st.pyplot(plt)
                        plt.clf()

                except Exception as e:
                    st.error('Could not compute SHAP: ' + str(e))

        st.markdown('---')
        st.write('Downloadable results:')
        csv = df_result.to_csv(index=False).encode('utf-8')
        st.download_button('Download predictions CSV', csv, file_name='predictions.csv')

else:
    st.info('Upload a CSV file to get started. You can use a subset of the Kaggle file for quick demo.')

st.sidebar.markdown('---')
st.sidebar.write('Developed for Fraud Detection')
