import pandas as pd
import plotly.graph_objects as go

def validate_input(text):
    """Validate input text"""
    if not isinstance(text, str):
        return False
    if len(text.strip()) == 0:
        return False
    return True

def create_metrics_chart(scores):
    """Create a radar chart for metrics visualization"""
    metrics = list(scores.keys())
    values = list(scores.values())

    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=metrics,
        fill='toself'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=False
    )

    return fig

def create_results_df(scores):
    """Create a DataFrame with results"""
    return pd.DataFrame({
        'Metric': list(scores.keys()),
        'Score': list(scores.values())
    })

def get_metric_descriptions():
    """Return descriptions for each metric"""
    return {
        'BLEU': """
        BLEU (Bilingual Evaluation Understudy) measures the precision of n-grams between 
        the candidate and reference translations. Scores range from 0 to 1, where 1 is perfect.
        """,

        'TER': """
        Translation Error Rate (TER) measures the number of edits needed to transform the 
        candidate into the reference translation. Lower scores are better, with 0 being perfect.
        """,

        'METEOR': """
        METEOR (Metric for Evaluation of Translation with Explicit ORdering) considers 
        exact matches between translations. Scores range from 0 to 1, where 1 is perfect.
        """,

        'BEER': """
        BEER (BEtter Evaluation as Ranking) is a trainable metric that combines different 
        features like character n-grams and word n-grams. For Cyrillic text, it uses 
        transliteration to Latin script. Scores range from 0 to 1, where 1 indicates a perfect match.
        """,

        'MAKTS': """
        Morpheme-Aware Kazakh Translation Score (MAKTS) is specifically designed for evaluating 
        translations involving the Kazakh language. It considers the agglutinative nature of 
        Kazakh by analyzing both root words and morphemes. The score ranges from 0 to 1, where:
        - Root words are given higher weight (70%)
        - Morpheme matches contribute 30% to the final score
        - 1 represents a perfect translation
        This metric is particularly useful for Kazakh language evaluation as it considers 
        the language's word formation patterns and morphological structure.
        """
    }