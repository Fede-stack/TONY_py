import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
from .LinguisticMarkers import *


class MarkersExtraction:
    def __init__(self, root):
        self.root = root
        root.title("Linguistic Marker Extractor")

        self.load_button = tk.Button(root, text="Select CSV input file", command=self.load_csv)
        self.load_button.pack(pady=5)
        self.input_label = tk.Label(root, text="No selected file")
        self.input_label.pack()

        tk.Label(root, text="Name of the column:").pack()
        self.column_entry = tk.Entry(root)
        self.column_entry.pack(pady=5)

        self.save_button = tk.Button(root, text="Choose the path and the name of the output .csv file:", command=self.save_csv)
        self.save_button.pack(pady=5)
        self.output_label = tk.Label(root, text="Selected None")
        self.output_label.pack()

        self.process_button = tk.Button(root, text="Extract and Export Markers", command=self.process)
        self.process_button.pack(pady=10)

        self.input_path = None
        self.output_path = None

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.input_path = path
            self.input_label.config(text=path)

    def save_csv(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv",
                                            filetypes=[("CSV files", "*.csv")])
        if path:
            self.output_path = path
            self.output_label.config(text=path)

    def process(self):
        if not self.input_path or not self.output_path:
            messagebox.showerror("Error", "You have to select both input and output file")
            return

        col = self.column_entry.get()
        if not col:
            messagebox.showerror("Error", "Insert the name of the text column!")
            return

        df = pd.read_csv(self.input_path)
        if col not in df.columns:
            messagebox.showerror("Error", f"Column '{col}' not found in CSV")
            return

        feature_extractor = LexiconLevelFeatures(language="en")

        results = []
        for idx, text in enumerate(df[col]):
            markers = feature_extractor.extract_markers(text)

            markers_dict = {
                # ── Index ──────────────────────────────────────────────
                "index": idx,

                # ── Lexical ────────────────────────────────────────────
                "lexical_diversity": markers.lexical_diversity,
                "word_prevalence": markers.word_prevalence,
                **{f"frequency_rare_{k}": v for k, v in markers.lexical_sophistication.items()},

                # ── Syntactic ──────────────────────────────────────────
                "sentence_complexity": markers.sentence_complexity,
                "subordination_rate": markers.subordination_rate,
                "coordination_rate": markers.coordination_rate,
                "mean_dependency_distance": markers.mean_dependency_distance,

                # ── Stylistic ──────────────────────────────────────────
                **{f"pronoun_{k}": v for k, v in markers.pronoun_usage.items()},
                **{f"tense_{k}": v for k, v in markers.verb_tense_distribution.items()},
                "negation_frequency": markers.negation_frequency,
                "question_ratio": markers.question_ratio,
                "exclamation_ratio": markers.exclamation_ratio,
                "incomplete_sentence_ratio": markers.incomplete_sentence_ratio,

                # ── Sentiment / Emotion ────────────────────────────────
                "sentiment_polarity": markers.sentiment_polarity,
                "sentiment_intensity": markers.sentiment_intensity,
                **{f"emotion_{k}": v for k, v in markers.emotion_scores.items()},
                **{f"affect_{k}": v for k, v in markers.affect_scores.items()},

                # ── Cohesion ───────────────────────────────────────────
                "cohesion_score": markers.cohesion_score,
                "lexical_overlap": markers.lexical_overlap,
                "connectives_usage": markers.connectives_usage,

                # ── Psychometric ───────────────────────────────────────
                **{f"cognitive_{k}": v for k, v in markers.cognitive_processes.items()},
                "social_processes": markers.social_processes,

                # ── Readability ────────────────────────────────────────
                "readability_index": markers.readability_index,
                "average_sentence_length": markers.average_sentence_length,

                # ── Graph (schizophrenia / psychosis) ──────────────────
                "graph_connectedness": markers.graph_connectedness,
                "semantic_coherence": markers.semantic_coherence,

                # ── NEW: Lexicon-based frequencies ─────────────────────
                "absolutist_word_frequency": markers.absolutist_word_frequency,
                "death_word_frequency": markers.death_word_frequency,
                "anxiety_word_frequency": markers.anxiety_word_frequency,
                "sadness_word_frequency": markers.sadness_word_frequency,
                "anger_word_frequency": markers.anger_word_frequency,
                "body_word_frequency": markers.body_word_frequency,
                "achievement_word_frequency": markers.achievement_word_frequency,

                # POS-based frequencies
                **{f"pos_{k}": v for k, v in markers.pos_frequencies.items()},
                **{f"extpos_{k}": v for k, v in markers.extended_pos.items()},
                **{f"morph_{k}": v for k, v in markers.morphological_features.items()},
                **{f"dep_{k}": v for k, v in markers.dependency_features.items()},
                **{f"ner_{k}": v for k, v in markers.ner_features.items()},

                # ── NEW: Temporal ──────────────────────────────────────
                "past_future_ratio": markers.past_future_ratio,

                # ── NEW: Structural ────────────────────────────────────
                "repetition_rate": markers.repetition_rate,
            }

            results.append(markers_dict)

        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_path, index=False)
        messagebox.showinfo("Success!", f"Results exported in {self.output_path}")
