import ipywidgets as widgets
from IPython.display import display, clear_output
import pandas as pd
from google.colab import files
import io
from .LinguisticMarkers import *

class MarkersExtractionColab:
    def __init__(self):
        # ── Upload input ────────────────────────────────────────
        self.upload_button = widgets.FileUpload(
            accept='.csv',
            multiple=False,
            description='📂 Input CSV'
        )

        # ── Nome colonna ────────────────────────────────────────
        self.column_entry = widgets.Text(
            placeholder='e.g. text',
            description='Column:',
            layout=widgets.Layout(width='300px')
        )

        # ── Nome output ─────────────────────────────────────────
        self.output_name = widgets.Text(
            value='output.csv',
            description='Output file:',
            layout=widgets.Layout(width='300px')
        )

        # ── Bottone process ─────────────────────────────────────
        self.process_button = widgets.Button(
            description='Extract and Export Markers',
            button_style='primary',
            icon='check'
        )
        self.process_button.on_click(self._process)

        # ── Output log ──────────────────────────────────────────
        self.out = widgets.Output()

        # ── Layout ──────────────────────────────────────────────
        display(widgets.VBox([
            widgets.HTML("<h3>🧠 TONY — Linguistic Marker Extractor</h3>"),
            self.upload_button,
            self.column_entry,
            self.output_name,
            self.process_button,
            self.out
        ]))

    def _process(self, b):
        with self.out:
            clear_output()

            # Validazione input
            if not self.upload_button.value:
                print("❌ No CSV file uploaded.")
                return

            col = self.column_entry.value.strip()
            if not col:
                print("❌ Insert the name of the text column!")
                return

            # Leggi CSV dall'upload
            uploaded_file = list(self.upload_button.value.values())[0]
            df = pd.read_csv(io.BytesIO(uploaded_file['content']))

            if col not in df.columns:
                print(f"❌ Column '{col}' not found. Available: {list(df.columns)}")
                return

            print(f"✅ File loaded: {len(df)} rows")
            print(f"⚙️  Extracting markers from column '{col}'...")

            feature_extractor = LexiconLevelFeatures(language="en")
            results = []

            for idx, text in enumerate(df[col]):
                if idx % 50 == 0:
                    print(f"   Processing row {idx}/{len(df)}...")

                markers = feature_extractor.extract_markers(text)
                markers_dict = {
                    "index": idx,
                    "lexical_diversity": markers.lexical_diversity,
                    "word_prevalence": markers.word_prevalence,
                    **{f"frequency_rare_{k}": v for k, v in markers.lexical_sophistication.items()},
                    "sentence_complexity": markers.sentence_complexity,
                    "subordination_rate": markers.subordination_rate,
                    "coordination_rate": markers.coordination_rate,
                    "mean_dependency_distance": markers.mean_dependency_distance,
                    **{f"pronoun_{k}": v for k, v in markers.pronoun_usage.items()},
                    **{f"tense_{k}": v for k, v in markers.verb_tense_distribution.items()},
                    "negation_frequency": markers.negation_frequency,
                    "question_ratio": markers.question_ratio,
                    "exclamation_ratio": markers.exclamation_ratio,
                    "incomplete_sentence_ratio": markers.incomplete_sentence_ratio,
                    "sentiment_polarity": markers.sentiment_polarity,
                    "sentiment_intensity": markers.sentiment_intensity,
                    **{f"emotion_{k}": v for k, v in markers.emotion_scores.items()},
                    **{f"affect_{k}": v for k, v in markers.affect_scores.items()},
                    "cohesion_score": markers.cohesion_score,
                    "lexical_overlap": markers.lexical_overlap,
                    "connectives_usage": markers.connectives_usage,
                    **{f"cognitive_{k}": v for k, v in markers.cognitive_processes.items()},
                    "social_processes": markers.social_processes,
                    "readability_index": markers.readability_index,
                    "average_sentence_length": markers.average_sentence_length,
                    "graph_connectedness": markers.graph_connectedness,
                    "semantic_coherence": markers.semantic_coherence,
                    "absolutist_word_frequency": markers.absolutist_word_frequency,
                    "death_word_frequency": markers.death_word_frequency,
                    "anxiety_word_frequency": markers.anxiety_word_frequency,
                    "sadness_word_frequency": markers.sadness_word_frequency,
                    "anger_word_frequency": markers.anger_word_frequency,
                    "body_word_frequency": markers.body_word_frequency,
                    "achievement_word_frequency": markers.achievement_word_frequency,
                    **{f"pos_{k}": v for k, v in markers.pos_frequencies.items()},
                    **{f"extpos_{k}": v for k, v in markers.extended_pos.items()},
                    **{f"morph_{k}": v for k, v in markers.morphological_features.items()},
                    **{f"dep_{k}": v for k, v in markers.dependency_features.items()},
                    **{f"ner_{k}": v for k, v in markers.ner_features.items()},
                    "past_future_ratio": markers.past_future_ratio,
                    "repetition_rate": markers.repetition_rate,
                }
                results.append(markers_dict)

            # Salva e scarica
            results_df = pd.DataFrame(results)
            output_filename = self.output_name.value.strip() or "output.csv"
            results_df.to_csv(output_filename, index=False)

            print(f"\n✅ Done! {len(results_df)} rows, {len(results_df.columns)} features extracted.")
            print(f"⬇️  Downloading '{output_filename}'...")
            files.download(output_filename)
