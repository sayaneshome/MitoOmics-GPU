
import argparse
import pandas as pd
import streamlit as st

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to results_summary.csv")
    args, _ = parser.parse_known_args()

    st.set_page_config(page_title="MitoOmics-GPU Dashboard", layout="wide")
    st.title("MitoOmics-GPU — Mito Health Index (MHI)")

    df = pd.read_csv(args.results)
    st.subheader("Subject-level MHI")
    st.dataframe(df.sort_values("MHI", ascending=False))

    if "subject_id" in df.columns:
        pick = st.selectbox("Subject", df["subject_id"].unique())
        row = df[df["subject_id"]==pick].T
        st.write("Selected subject details:")
        st.dataframe(row)

    st.caption("Research-use only. Not for diagnosis.")

if __name__ == "__main__":
    main()
