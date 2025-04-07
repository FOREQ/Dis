import streamlit as st
import pandas as pd
from metrics import (
    calculate_bleu, calculate_ter, calculate_meteor,
    calculate_makts, calculate_beer
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
using multiple metrics: BLEU, TER, METEOR, BEER, and MAKTS (Morpheme-Aware Kazakh Translation Score).
""")

# Language pair selection
language_pairs = [
    "English → Kazakh",
    "Russian → Kazakh",
    "Kazakh → English",
    "Kazakh → Russian"
]
selected_pair = st.selectbox("Select Language Pair", language_pairs)

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
        reference_text = st.text_area("Enter reference translation", height=150)

    with col2:
        st.subheader("Candidate Translation")
        candidate_text = st.text_area("Enter candidate translation", height=150)

else:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Reference Translation File")
        reference_file = st.file_uploader("Upload reference translation", type=['txt'])
        if reference_file:
            reference_text = reference_file.getvalue().decode("utf-8")

    with col2:
        st.subheader("Candidate Translation File")
        candidate_file = st.file_uploader("Upload candidate translation", type=['txt'])
        if candidate_file:
            candidate_text = candidate_file.getvalue().decode("utf-8")

# Evaluate button
if st.button("Evaluate Translation"):
    if not validate_input(reference_text) or not validate_input(candidate_text):
        st.error("Please provide both reference and candidate translations")
    else:
        with st.spinner("Calculating metrics..."):
            # Calculate scores
            bleu_score = calculate_bleu(reference_text, candidate_text, selected_pair.split()[0].lower())
            ter_score = calculate_ter(reference_text, candidate_text)
            meteor_score = calculate_meteor(reference_text, candidate_text)
            beer_score = calculate_beer(reference_text, candidate_text)

            # Add MAKTS score for translations involving Kazakh
            makts_score = 0.0
            if "Kazakh" in selected_pair:
                makts_score = calculate_makts(reference_text, candidate_text)

            scores = {
                'BLEU': bleu_score,
                'TER': ter_score,
                'METEOR': meteor_score,
                'BEER': beer_score,
                'MAKTS': makts_score if "Kazakh" in selected_pair else None
            }

            # Remove None values for non-Kazakh translations
            scores = {k: v for k, v in scores.items() if v is not None}

            # Display results
            st.subheader("Results")

            # Create three columns for the layout
            col1, col2 = st.columns([2, 3])

            with col1:
                # Display metrics table
                results_df = create_results_df(scores)
                st.dataframe(results_df, hide_index=True)

                # Export results button
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Results CSV",
                    csv,
                    "translation_metrics.csv",
                    "text/csv",
                    key='download-csv'
                )

            with col2:
                # Display radar chart
                st.plotly_chart(create_metrics_chart(scores), use_container_width=True)

            # Display metric descriptions
            st.subheader("Metric Descriptions")
            descriptions = get_metric_descriptions()

            for metric, desc in descriptions.items():
                with st.expander(f"About {metric}"):
                    st.write(desc)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Translation Quality Assessment Tool - 2024</p>
</div>
""", unsafe_allow_html=True)