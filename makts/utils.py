import pandas as pd
import plotly.graph_objects as go
import numpy as np # Import numpy for safe handling of potential NaN/inf

def validate_input(text):
    """Validate input text"""
    if not isinstance(text, str):
        return False
    # Check if text is empty or contains only whitespace
    if not text.strip():
        return False
    return True

def create_metrics_chart(scores):
    """Create a radar chart for metrics visualization"""
    # Filter out non-numeric or unsuitable values before plotting
    metrics = []
    values = []
    for k, v in scores.items():
        # Ensure value is numeric and finite. Handle TER separately if needed.
        if isinstance(v, (int, float)) and np.isfinite(v):
             # Special handling for TER: invert it for the chart (higher is better)
             # Or cap it if it's too high and makes the chart unreadable
            if k == 'TER':
                 # Invert TER: 1 - TER (closer to 1 is better). Ensure score >= 0.
                 chart_value = max(0, 1 - v)
                 # Add TER (Inv) to the label for clarity
                 metrics.append(f"{k} (Inv)")
                 values.append(chart_value)
            elif 0 <= v <= 1: # Keep scores already in 0-1 range
                metrics.append(k)
                values.append(v)
            # Add other scaling logic if needed for other metrics not in 0-1

    if not metrics: # If no valid metrics to plot
        return go.Figure() # Return empty figure


    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=metrics,
        fill='toself',
        # Optional: Add text labels on points
        # mode = 'lines+markers+text',
        # text = [f'{v:.2f}' for v in values],
        # textposition = 'top center'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1], # Ensure axis is 0 to 1
                tickvals=[0, 0.25, 0.5, 0.75, 1], # Define ticks
                ticktext=['0', '0.25', '0.5', '0.75', '1'] # Define tick labels
            )),
        showlegend=False,
         # Add title to the chart
        title=dict(
            text='Translation Quality Metrics Overview',
            x=0.5 # Center title
        ),
        # Adjust margin if text labels overlap
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig


def create_results_df(scores):
    """Create a DataFrame with results, formatting scores"""
    formatted_scores = []
    for score in scores.values():
        if isinstance(score, (int, float)):
             # Format floats to 4 decimal places, keep integers as is
             formatted_scores.append(f"{score:.4f}" if isinstance(score, float) else str(score))
        else:
             formatted_scores.append(str(score)) # Keep non-numeric as string

    return pd.DataFrame({
        'Metric': list(scores.keys()),
        'Score': formatted_scores # Use formatted scores
    })


def get_metric_descriptions():
    """Return descriptions for each metric"""
    # Use markdown for better formatting control within Streamlit expanders
    return {
        'BLEU': """
        **BLEU (Bilingual Evaluation Understudy)** measures the *precision* of n-grams (sequences of n words) in the candidate translation compared to the reference translation(s).
        * **Range:** 0 to 1 (or 0 to 100).
        * **Higher is better.**
        * It correlates well with human judgment on average but can penalize valid translations that use different wording. It favors shorter translations due to its brevity penalty calculation.
        """,

        'chrF': """
        **chrF (character n-gram F-score)** measures the overlap of character n-grams (sequences of n characters) between the candidate and reference translations. It computes precision and recall based on these character n-grams and then calculates their F-score (harmonic mean).
        * **Range:** 0 to 1 (or 0 to 100).
        * **Higher is better.**
        * It is less sensitive to tokenization issues than word-based metrics like BLEU and often correlates better with human judgment for morphologically rich languages or when dealing with translation noise.
        """, # <<< ДОБАВЛЕНО ОПИСАНИЕ chrF

        'TER': """
        **TER (Translation Error Rate)** measures the minimum number of edits (insertions, deletions, substitutions, and shifts) required to change the candidate translation to match the reference translation, normalized by the reference length.
        * **Range:** 0 to infinity (often capped around 1.0 or 2.0 for comparison).
        * **Lower is better (0 is perfect).**
        * It directly reflects the post-editing effort needed.
        """,

        'METEOR': """
        **METEOR (Metric for Evaluation of Translation with Explicit ORdering)** calculates a score based on aligned unigrams (single words) between the candidate and reference translations. It considers exact matches, stemmed matches, and synonym matches (though simpler versions might only use exact matches). It uses precision, recall, and a penalty for incorrect word order (chunkiness).
        * **Range:** 0 to 1.
        * **Higher is better.**
        * Generally correlates better with human judgment than BLEU, especially at the sentence level.
        """,

        'BEER': """
        **BEER (BEtter Evaluation as Ranking)** is a trainable metric (though often used with default parameters) that combines various features, including character n-grams, word n-grams, and optionally information about word order or syntax. The provided version focuses on character n-grams and word overlap with transliteration for Cyrillic.
        * **Range:** 0 to 1.
        * **Higher is better.**
        * Designed to be robust and potentially adaptable through training.
        """,

        'MAKTS': """
        **MAKTS (Morpheme-Aware Kazakh Translation Score)** is a custom metric designed for evaluating translations involving the morphologically rich Kazakh language. It segments words into stems and suffixes (morphemes) and calculates a similarity score based on how well the morphological structure matches between the reference and candidate words.
        * **Range:** 0 to 1.
        * **Higher is better.**
        * Aims to capture translation quality aspects specific to agglutinative languages like Kazakh, which might be missed by purely surface-level metrics. The weighting between stem and suffix similarity can be adjusted.
        """
    }