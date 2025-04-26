import streamlit as st
import pandas as pd
from metrics import (
    calculate_bleu, calculate_ter, calculate_meteor,
    calculate_makts, calculate_beer, calculate_chrf # <<< ДОБАВЛЕН calculate_chrf
)
from utils import validate_input, create_metrics_chart, create_results_df, get_metric_descriptions

# Page configuration
st.set_page_config(
    page_title="Translation Quality Assessment",
    layout="wide"
)

# Title and description
st.title("Translation Quality Assessment Tool")
st.markdown("""
This tool helps evaluate translation quality between English, Russian, and Kazakh languages
using multiple metrics: BLEU, chrF, TER, METEOR, BEER, and MAKTS (Morpheme-Aware Kazakh Translation Score).
""") # <<< ДОБАВЛЕН chrF в описание

# Language pair selection
language_pairs = [
    "English → Kazakh",
    "Russian → Kazakh",
    "Kazakh → English",
    "Kazakh → Russian"
]
# Determine default language based on selected pair for BLEU calculation
# We need the source language for potential NLTK tokenization
selected_pair = st.selectbox("Select Language Pair", language_pairs)
source_language = selected_pair.split(" → ")[0].lower()


# Input method selection
input_method = st.radio(
    "Choose input method",
    ["Direct Text Input", "File Upload"]
)

reference_text = ""
candidate_text = ""

if input_method == "Direct Text Input":
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Reference Translation")
        reference_text = st.text_area("Enter reference translation", height=150, key="ref_text")

    with col2:
        st.subheader("Candidate Translation")
        candidate_text = st.text_area("Enter candidate translation", height=150, key="cand_text")

else: # File Upload
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Reference Translation File")
        reference_file = st.file_uploader("Upload reference translation", type=['txt'], key="ref_file")
        if reference_file:
            try:
                reference_text = reference_file.getvalue().decode("utf-8")
                st.text_area("Reference Content:", reference_text, height=100, disabled=True) # Display content
            except Exception as e:
                st.error(f"Error reading reference file: {e}")


    with col2:
        st.subheader("Candidate Translation File")
        candidate_file = st.file_uploader("Upload candidate translation", type=['txt'], key="cand_file")
        if candidate_file:
            try:
                candidate_text = candidate_file.getvalue().decode("utf-8")
                st.text_area("Candidate Content:", candidate_text, height=100, disabled=True) # Display content
            except Exception as e:
                 st.error(f"Error reading candidate file: {e}")


# Evaluate button
if st.button("Evaluate Translation"):
    if not validate_input(reference_text) or not validate_input(candidate_text):
        st.error("Please provide both reference and candidate translations (either via text or file upload).")
    else:
        with st.spinner("Calculating metrics..."):
            # Calculate scores
            # Pass the determined source language to calculate_bleu
            bleu_score = calculate_bleu(reference_text, candidate_text, source_language)
            # chrF does not require language parameter for sacrebleu implementation
            chrf_score = calculate_chrf(reference_text, candidate_text) # <<< ВЫЧИСЛЕНИЕ chrF
            ter_score = calculate_ter(reference_text, candidate_text)
            meteor_score = calculate_meteor(reference_text, candidate_text)
            beer_score = calculate_beer(reference_text, candidate_text)

            # Add MAKTS score for translations involving Kazakh
            makts_score = None # Initialize as None
            if "kazakh" in selected_pair.lower(): # Check if Kazakh is involved
                makts_score = calculate_makts(reference_text, candidate_text)

            scores = {
                'BLEU': bleu_score,
                'chrF': chrf_score, # <<< ДОБАВЛЕН chrF в словарь
                'TER': ter_score,
                'METEOR': meteor_score,
                'BEER': beer_score,
                'MAKTS': makts_score # Keep MAKTS even if None initially
            }

            # Remove None values (e.g., MAKTS if not Kazakh) before displaying
            # Also filter TER if it's excessively high (e.g., > 2.0) as it might skew radar chart
            scores_filtered = {k: v for k, v in scores.items() if v is not None}
            scores_for_chart = {k: v for k, v in scores_filtered.items() if k != 'TER' or (k == 'TER' and v <= 2.0)}


            # Display results
            st.subheader("Results")

            # Create two columns for the layout
            col_res1, col_res2 = st.columns([2, 3]) # Adjusted ratio if needed

            with col_res1:
                # Display metrics table (using filtered scores)
                results_df = create_results_df(scores_filtered) # Show all calculated scores in table
                st.dataframe(results_df, hide_index=True, use_container_width=True)


                # Export results button
                try:
                    csv = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results CSV", # Changed label for clarity
                        data=csv,
                        file_name="translation_metrics.csv", # Corrected filename
                        mime="text/csv", # Corrected mime type
                        key='download-csv'
                    )
                except Exception as e:
                    st.error(f"Failed to generate CSV: {e}")


            with col_res2:
                # Display radar chart (using scores suitable for chart)
                if scores_for_chart: # Ensure there are scores to plot
                    st.plotly_chart(create_metrics_chart(scores_for_chart), use_container_width=True)
                else:
                    st.warning("No suitable scores to display in the chart.")


            # Display metric descriptions
            st.subheader("Metric Descriptions")
            # Fetch descriptions, including the new chrF one from utils.py
            descriptions = get_metric_descriptions()


            # Display descriptions for the metrics actually calculated
            for metric in scores_filtered.keys():
                 if metric in descriptions:
                     with st.expander(f"About {metric}"):
                         st.markdown(descriptions[metric]) # Use markdown for better formatting


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: grey;'>
    <p>Translation Quality Assessment Tool - 2024</p>
</div>
""", unsafe_allow_html=True)