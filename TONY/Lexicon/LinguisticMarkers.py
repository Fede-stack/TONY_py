import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import Counter
import string
from wordfreq import word_frequency
import spacy
from nrclex import NRCLex
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

nltk.download('punkt_tab')
nltk.download('vader_lexicon', quiet=True)

nlp = spacy.load("en_core_web_sm")


class MentalHealthCondition(Enum):
    """Enumeration of supported mental health conditions"""
    DEPRESSION = "depression"
    ANXIETY = "anxiety"
    BIPOLAR = "bipolar"
    PTSD = "ptsd"
    SCHIZOPHRENIA = "schizophrenia"
    ADHD = "adhd"
    EATING_DISORDER = "eating_disorder"
    STRESS = "stress"
    SUICIDE_RISK = "suicide_risk"
    OCD = "ocd"


@dataclass
class LinguisticMarkers:
    """Data structure to contain extracted linguistic markers"""
    # Lexical markers
    lexical_diversity: float
    lexical_sophistication: float
    word_prevalence: float

    # Syntactic markers
    sentence_complexity: float
    subordination_rate: float
    coordination_rate: float

    # Stylistic markers
    pronoun_usage: Dict[str, float]
    verb_tense_distribution: Dict[str, float]
    negation_frequency: float

    # Semantic markers
    emotion_scores: Dict[str, float]
    sentiment_polarity: float
    sentiment_intensity: float

    # Cohesion markers
    cohesion_score: float
    lexical_overlap: float
    connectives_usage: float

    # Psychometric markers
    affect_scores: Dict[str, float]
    cognitive_processes: Dict[str, float]
    social_processes: float

    # Readability markers
    readability_index: float
    average_sentence_length: float

    # Graph markers (for psychosis/schizophrenia)
    graph_connectedness: Optional[float] = None
    semantic_coherence: Optional[float] = None

    # Lexicon-based frequencies (lexicons already existed but were never extracted)
    absolutist_word_frequency: float = 0.0      # depression, anxiety, OCD
    death_word_frequency: float = 0.0           # suicide risk
    anxiety_word_frequency: float = 0.0         # anxiety
    sadness_word_frequency: float = 0.0         # depression
    anger_word_frequency: float = 0.0           # anger / PTSD

    # Syntactic / structural
    question_ratio: float = 0.0                 # proportion of sentences ending with '?'
    exclamation_ratio: float = 0.0              # proportion of sentences ending with '!'
    incomplete_sentence_ratio: float = 0.0      # sentences < 3 words (thought disorganization)
    mean_dependency_distance: float = 0.0       # syntactic complexity via spaCy dep tree

    # Semantic / temporal
    past_future_ratio: float = 0.0              # past / (future + epsilon) — depression vs anxiety
    repetition_rate: float = 0.0                # word repetition rate (OCD, rumination)

    # Domain-specific lexicon frequencies
    body_word_frequency: float = 0.0            # somatic symptoms (depression, eating disorder)
    achievement_word_frequency: float = 0.0     # grandiosity (bipolar/mania)

    # POS-based frequencies
    pos_frequencies: Dict[str, float] = None         # prep, auxverb, adverb, conj
    extended_pos: Dict[str, float] = None             # noun, verb, adjective, interjection
    morphological_features: Dict[str, float] = None  # indicative, subjunctive, singular, plural
    dependency_features: Dict[str, float] = None     # nsubj_rate, dobj_rate
    ner_features: Dict[str, float] = None            # person_ref_rate, temporal_ref_rate


class LexiconLevelFeatures:
    """
    Class for extracting linguistic markers for various mental health conditions.

    Based on:
    - Natural Language Processing for mental health interventions
    - Multimodal approaches (text + optional acoustic features)
    """

    def __init__(self, language: str = "en"):
        self.language = language
        self.condition_specific_markers = self._initialize_condition_markers()
        self._load_lexicons()
        self._insight_lemmas = set(self.insight_words)
        self._causation_lemmas = set(self.causation_words)
        self._certainty_lemmas = set(self.certainty_words)
        self._tentative_lemmas = set(self.tentative_words)
        self._social_lemmas = set(self.social_words)

    def _load_lexicons(self):
        """Load linguistic lexicons and dictionaries for marker extraction"""
        self.first_person_pronouns = {
            'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'
        }
        self.second_person_pronouns = {
            'you', 'your', 'yours', 'yourself', 'yourselves'
        }
        self.third_person_pronouns = {
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves'
        }
        self.negation_words = {
            'not', 'no', 'never', 'nothing', 'none', 'nobody', 'nowhere',
            'neither', 'nor', "n't", 'cannot', 'cant', "won't", "don't",
            "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't"
        }
        self.absolutist_words = {
            'always', 'never', 'everything', 'nothing', 'everyone', 'nobody',
            'all', 'none', 'every', 'completely', 'absolutely', 'totally',
            'entirely', 'forever', 'constant', 'permanent'
        }
        self.positive_emotion_words = {
            'happy', 'joy', 'love', 'good', 'great', 'excellent', 'wonderful',
            'amazing', 'fantastic', 'beautiful', 'pleasant', 'excited', 'cheerful',
            'delighted', 'glad', 'pleased', 'content', 'satisfied', 'grateful'
        }
        self.negative_emotion_words = {
            'sad', 'depressed', 'unhappy', 'miserable', 'terrible', 'awful',
            'bad', 'horrible', 'anxious', 'worried', 'afraid', 'scared',
            'angry', 'mad', 'frustrated', 'upset', 'hurt', 'pain', 'lonely'
        }
        self.anxiety_words = {
            'worried', 'anxious', 'nervous', 'tense', 'stress', 'fear',
            'afraid', 'scared', 'panic', 'worry', 'concern', 'uneasy',
            'apprehensive', 'dread', 'frightened'
        }
        self.sadness_words = {
            'sad', 'depressed', 'down', 'blue', 'unhappy', 'miserable',
            'hopeless', 'helpless', 'worthless', 'empty', 'lonely', 'crying',
            'tears', 'grief', 'sorrow', 'despair'
        }
        self.anger_words = {
            'angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated',
            'rage', 'hate', 'hostile', 'bitter', 'resentful', 'outraged'
        }
        self.death_words = {
            'death', 'die', 'dead', 'suicide', 'kill', 'end', 'gone',
            'disappear', 'cease', 'funeral', 'grave', 'coffin'
        }
        self.insight_words = {
            'think', 'know', 'understand', 'realize', 'believe', 'feel',
            'consider', 'recognize', 'wonder', 'imagine', 'remember', 'suspect',
            'doubt', 'agree', 'mean', 'guess', 'assume', 'perceive', 'recall'
        }
        self.causation_words = {
            'because', 'cause', 'since', 'therefore', 'thus', 'hence',
            'consequently', 'due', 'reason', 'so', 'for',
            'accordingly', 'henceforth'
        }
        self.certainty_words = {
            'always', 'never', 'certainly', 'definitely', 'obviously',
            'clearly', 'undoubtedly', 'sure', 'certain', 'indeed', 'surely',
            'necessarily', 'plainly', 'indubitably', 'unquestionably'
        }
        self.tentative_words = {
            'maybe', 'perhaps', 'possibly', 'might', 'could', 'probably',
            'seem', 'appear', 'guess', 'suppose', 'uncertain', 'likely',
            'tend', 'suggest', 'indicate', 'imply', 'potentially'
        }
        self.social_words = {
            'talk', 'share', 'friend', 'family', 'people', 'together',
            'social', 'community', 'relationship', 'connect', 'meet',
            'girlfriend', 'boyfriend'
        }
        self.past_tense_markers = {'was', 'were', 'had', 'did', 'been'}
        self.future_tense_markers = {
            'will', 'shall', 'going to', 'gonna', 'would', 'could', 'might', 'may', "'ll"
        }
        self.subordinators = {
            'that', 'which', 'who', 'whom', 'whose', 'when', 'where',
            'if', 'because', 'although', 'though', 'while', 'since',
            'unless', 'until', 'before', 'after', 'as'
        }
        self.coordinators = {'and', 'or', 'but', 'nor', 'yet', 'so', 'for'}
        self.connectives = {
            'however', 'therefore', 'moreover', 'furthermore', 'nevertheless',
            'consequently', 'additionally', 'meanwhile', 'thus', 'hence',
            'besides', 'otherwise', 'instead', 'then', 'next', 'finally'
        }


        # Somatic / body words (relevant for depression, eating disorders)
        self.body_words = {
            'sleep', 'tired', 'exhausted', 'hungry', 'eat', 'food', 'appetite',
            'weight', 'body', 'pain', 'ache', 'headache', 'stomach', 'sick',
            'nausea', 'fatigue', 'energy', 'rest', 'insomnia', 'chest', 'heart',
            'breathe', 'sweat', 'shake', 'tremble', 'dizzy', 'throat'
        }

        # Achievement / grandiosity words (relevant for mania/bipolar)
        self.achievement_words = {
            'achieve', 'success', 'win', 'best', 'genius', 'special', 'powerful',
            'superior', 'incredible', 'great', 'amazing', 'perfect', 'outstanding',
            'brilliant', 'exceptional', 'unique', 'talented', 'gifted', 'chosen',
            'mission', 'destiny', 'leader', 'important', 'famous', 'rich', 'unlimited'
        }

    def _initialize_condition_markers(self) -> Dict[MentalHealthCondition, List[str]]:
        return {
            MentalHealthCondition.DEPRESSION: [
                "lexical_diversity", "first_person_pronouns", "negative_emotions",
                "past_tense", "absolutist_words", "social_processes_low",
                "sadness_word_frequency", "body_word_frequency", "past_future_ratio_high"
            ],
            MentalHealthCondition.ANXIETY: [
                "future_tense", "fear_words", "cognitive_processes_high",
                "tentative_language", "arousal_high", "sentence_complexity_high",
                "anxiety_word_frequency", "question_ratio", "mean_dependency_distance"
            ],
            MentalHealthCondition.BIPOLAR: [
                "lexical_variety_fluctuation", "positive_emotion_episodes",
                "activity_words", "flight_of_ideas", "grandiose_language",
                "achievement_word_frequency", "exclamation_ratio"
            ],
            MentalHealthCondition.PTSD: [
                "trauma_related_words", "hypervigilance_markers",
                "avoidance_language", "emotional_numbing", "intrusive_thoughts",
                "anger_word_frequency", "past_future_ratio_high"
            ],
            MentalHealthCondition.SCHIZOPHRENIA: [
                "graph_connectedness_low", "loosening_associations",
                "neologisms", "semantic_incoherence", "reduced_speech_connectivity",
                "incomplete_sentence_ratio", "repetition_rate"
            ],
            MentalHealthCondition.ADHD: [
                "stylistic_features", "lexical_richness", "cohesion_markers",
                "impulsive_language", "attention_shifts",
                "incomplete_sentence_ratio", "repetition_rate"
            ],
            MentalHealthCondition.STRESS: [
                "negative_affect", "worry_words", "time_pressure_words", "coping_language",
                "anxiety_word_frequency", "body_word_frequency"
            ],
            MentalHealthCondition.SUICIDE_RISK: [
                "hopelessness_markers", "death_related_words", "isolation_language",
                "burden_perception", "lack_of_future_references",
                "death_word_frequency", "sadness_word_frequency", "past_future_ratio_high"
            ],
            MentalHealthCondition.OCD: [
                "repetitive_language", "certainty_words", "absolutist_words",
                "cognitive_processes_high", "negation_frequency",
                "absolutist_word_frequency", "repetition_rate"
            ],
        }

    def extract_markers(self, text: str,
                        condition: Optional[MentalHealthCondition] = None) -> LinguisticMarkers:
        """Extract linguistic markers from text."""
        markers = LinguisticMarkers(
            lexical_diversity=self._compute_lexical_diversity(text),
            lexical_sophistication=self._compute_lexical_sophistication(text),
            word_prevalence=self._compute_word_prevalence(text),
            sentence_complexity=self._compute_sentence_complexity(text),
            subordination_rate=self._compute_subordination_rate(text),
            coordination_rate=self._compute_coordination_rate(text),
            pronoun_usage=self._extract_pronoun_usage(text),
            verb_tense_distribution=self._extract_verb_tense(text),
            negation_frequency=self._compute_negation_frequency(text),
            emotion_scores=self._extract_emotions(text),
            sentiment_polarity=self._compute_sentiment_polarity(text),
            sentiment_intensity=self._compute_sentiment_intensity(text),
            cohesion_score=self._compute_cohesion(text),
            lexical_overlap=self._compute_lexical_overlap(text),
            connectives_usage=self._compute_connectives(text),
            affect_scores=self._extract_affect_scores(text),
            cognitive_processes=self._extract_cognitive_processes(text),
            social_processes=self._extract_social_processes(text),
            readability_index=self._compute_readability(text),
            average_sentence_length=self._compute_avg_sentence_length(text),
        )

        markers.graph_connectedness = self._compute_graph_connectedness(text)
        markers.semantic_coherence = self._compute_semantic_coherence(text)

        # --- NEW FEATURES ---
        markers.absolutist_word_frequency = self._compute_lexicon_frequency(text, self.absolutist_words)
        markers.death_word_frequency = self._compute_lexicon_frequency(text, self.death_words)
        markers.anxiety_word_frequency = self._compute_lexicon_frequency(text, self.anxiety_words)
        markers.sadness_word_frequency = self._compute_lexicon_frequency(text, self.sadness_words)
        markers.anger_word_frequency = self._compute_lexicon_frequency(text, self.anger_words)
        markers.body_word_frequency = self._compute_lexicon_frequency(text, self.body_words)
        markers.achievement_word_frequency = self._compute_lexicon_frequency(text, self.achievement_words)

        markers.question_ratio = self._compute_punctuation_sentence_ratio(text, '?')
        markers.exclamation_ratio = self._compute_punctuation_sentence_ratio(text, '!')
        markers.incomplete_sentence_ratio = self._compute_incomplete_sentence_ratio(text)
        markers.mean_dependency_distance = self._compute_mean_dependency_distance(text)

        markers.past_future_ratio = self._compute_past_future_ratio(markers.verb_tense_distribution)
        markers.repetition_rate = self._compute_repetition_rate(text)
        markers.pos_frequencies        = self._compute_pos_frequencies(text)
        markers.extended_pos           = self._compute_extended_pos(text)
        markers.morphological_features = self._compute_morphological_features(text)
        markers.dependency_features    = self._compute_dependency_features(text)
        markers.ner_features           = self._compute_ner_features(text)

        return markers


    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        words = text.split()
        return [w for w in words if w]

    def _lemmatize_words(self, text: str) -> List[str]:
        doc = nlp(text)
        return [t.lemma_.lower() for t in doc if t.is_alpha]

    def _get_sentences(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_lexical_diversity(self, text: str) -> float:
        words = self._tokenize(text)
        if not words:
            return 0.0
        return len(set(words)) / len(words)

    def _compute_lexical_sophistication(self, text: str) -> Dict[str, float]:
        words = self._tokenize(text)
        if not words:
            return {"mean_frequency": 0.0, "std_frequency": 0.0}
        freqs = np.array([word_frequency(w, self.language) for w in words])
        return {
            "mean_frequency": round(float(np.mean(freqs)), 6),
            "std_frequency":  round(float(np.std(freqs)), 6),
        }

    def _compute_word_prevalence(self, text: str) -> float:
        words = self._tokenize(text)
        if not words:
            return 0.0
        common_word_count = sum(
            1 for word in words
            if word in self.first_person_pronouns
            or word in self.second_person_pronouns
            or word in self.third_person_pronouns
            or word in self.coordinators
            or word in self.subordinators
        )
        return common_word_count / len(words)

    def _compute_sentence_complexity(self, text: str) -> float:
        sentences = self._get_sentences(text)
        if not sentences:
            return 0.0
        complexities = []
        for sentence in sentences:
            words = self._tokenize(sentence)
            if not words:
                continue
            sub = sum(1 for w in words if w in self.subordinators)
            coo = sum(1 for w in words if w in self.coordinators)
            complexities.append((sub * 2 + coo) / len(words))
        return np.mean(complexities) if complexities else 0.0

    def _compute_subordination_rate(self, text: str) -> float:
        words = self._tokenize(text)
        if not words:
            return 0.0
        return sum(1 for w in words if w in self.subordinators) / len(words)

    def _compute_coordination_rate(self, text: str) -> float:
        words = self._tokenize(text)
        if not words:
            return 0.0
        return sum(1 for w in words if w in self.coordinators) / len(words)

    def _extract_pronoun_usage(self, text: str) -> Dict[str, float]:
        words = self._tokenize(text)
        if not words:
            return {"first_person": 0.0, "second_person": 0.0, "third_person": 0.0}
        total = len(words)
        return {
            "first_person":  sum(1 for w in words if w in self.first_person_pronouns)  / total,
            "second_person": sum(1 for w in words if w in self.second_person_pronouns) / total,
            "third_person":  sum(1 for w in words if w in self.third_person_pronouns)  / total,
        }

    def _extract_verb_tense(self, text: str) -> Dict[str, float]:
        doc = nlp(text)
        tenses = Counter({"past": 0, "present": 0, "future": 0})
        for token in doc:
            if token.pos_ in ("VERB", "AUX"):
                tense = token.morph.get("Tense")
                if "Past" in tense:
                    tenses["past"] += 1
                elif "Pres" in tense:
                    tenses["present"] += 1
                elif "Fut" in tense:
                    tenses["future"] += 1
        total = sum(tenses.values())
        if total == 0:
            return {"past": 0.0, "present": 0.0, "future": 0.0}
        return {k: round(v / total, 4) for k, v in tenses.items()}

    def _compute_negation_frequency(self, text: str) -> float:
        negation_words = {"not", "no", "never", "nothing", "none", "non"}
        words = text.lower().split()
        return sum(1 for w in words if w in negation_words) / len(words) if words else 0.0

    def _extract_emotions(self, text: str) -> Dict[str, float]:
        target_emotions = ["joy", "sadness", "anger", "fear", "disgust",
                           "surprise", "anticipation", "trust"]
        if not text.strip():
            return {emo: 0.0 for emo in target_emotions}
        nrc = NRCLex(text)
        raw = nrc.raw_emotion_scores
        filtered = {emo: raw.get(emo, 0) for emo in target_emotions}
        total = sum(filtered.values())
        if total == 0:
            return {emo: 0.0 for emo in target_emotions}
        return {emo: round(v / total, 4) for emo, v in filtered.items()}

    def _compute_sentiment_polarity(self, text: str) -> float:
        words = self._tokenize(text)
        if not words or not text.strip():
            return 0.0
        nrc = NRCLex(text)
        raw = nrc.raw_emotion_scores
        pos = raw.get('positive', 0)
        neg = raw.get('negative', 0)
        total = pos + neg
        return (pos - neg) / total if total else 0.0

    def _compute_sentiment_intensity(self, text: str) -> float:
        if not text or not text.strip():
            return 0.0
        scores = SentimentIntensityAnalyzer().polarity_scores(text)
        return round(scores["compound"], 4)

    def _compute_cohesion(self, text: str) -> float:
        sentences = self._get_sentences(text)
        if len(sentences) < 2:
            return 1.0
        scores = []
        for i in range(len(sentences) - 1):
            w1 = set(self._tokenize(sentences[i]))
            w2 = set(self._tokenize(sentences[i + 1]))
            if not w1 or not w2:
                continue
            union = len(w1 | w2)
            if union:
                scores.append(len(w1 & w2) / union)
        return np.mean(scores) if scores else 0.0

    def _compute_lexical_overlap(self, text: str) -> float:
        sentences = self._get_sentences(text)
        if len(sentences) < 2:
            return 0.0
        overlaps = []
        for i in range(len(sentences) - 1):
            w1 = self._tokenize(sentences[i])
            w2 = self._tokenize(sentences[i + 1])
            if not w1 or not w2:
                continue
            common = len(set(w1) & set(w2))
            overlaps.append(common / min(len(w1), len(w2)))
        return np.mean(overlaps) if overlaps else 0.0

    def _compute_connectives(self, text: str) -> float:
        words = self._tokenize(text)
        if not words:
            return 0.0
        return sum(1 for w in words if w in self.connectives) / len(words)

    def _extract_affect_scores(self, text: str) -> Dict[str, float]:
        if not text.strip():
            return {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
        nrc = NRCLex(text)
        emotions = nrc.raw_emotion_scores
        pos   = emotions.get("positive", 0)
        neg   = emotions.get("negative", 0)
        anger = emotions.get("anger", 0)
        fear  = emotions.get("fear", 0)
        joy   = emotions.get("joy", 0)
        total = sum(emotions.values()) or 1
        return {
            "valence":   round((pos - neg) / total, 4),
            "arousal":   round((anger + fear + joy) / total, 4),
            "dominance": round(1 - fear / total, 4),
        }

    def _extract_cognitive_processes(self, text: str) -> Dict[str, float]:
        lemmas = self._lemmatize_words(text)
        if not lemmas:
            return {"insight": 0.0, "causation": 0.0, "certainty": 0.0, "tentative": 0.0}
        total = len(lemmas)
        return {
            "insight":   sum(1 for w in lemmas if w in self._insight_lemmas)   / total,
            "causation": sum(1 for w in lemmas if w in self._causation_lemmas) / total,
            "certainty": sum(1 for w in lemmas if w in self._certainty_lemmas) / total,
            "tentative": sum(1 for w in lemmas if w in self._tentative_lemmas) / total,
        }

    def _extract_social_processes(self, text: str) -> float:
        lemmas = self._lemmatize_words(text)
        if not lemmas:
            return 0.0
        return sum(1 for w in lemmas if w in self._social_lemmas)

    def _compute_readability(self, text: str) -> float:
        sentences = self._get_sentences(text)
        words = self._tokenize(text)
        if not sentences or not words:
            return 0.0

        def count_syllables(word):
            vowels = 'aeiou'
            word = word.lower()
            count = 0
            prev_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            if word.endswith('e'):
                count -= 1
            return max(1, count)

        total_syllables = sum(count_syllables(w) for w in words)
        avg_sent_len = len(words) / len(sentences)
        avg_syl_per_word = total_syllables / len(words)
        flesch = 206.835 - 1.015 * avg_sent_len - 84.6 * avg_syl_per_word
        return max(0.0, min(flesch / 100.0, 1.0))

    def _compute_avg_sentence_length(self, text: str) -> float:
        sentences = self._get_sentences(text)
        if not sentences:
            return 0.0
        return np.mean([len(self._tokenize(s)) for s in sentences])

    def _compute_graph_connectedness(self, text: str) -> float:
        sentences = self._get_sentences(text)
        if len(sentences) < 2:
            return 1.0
        word_connections: Dict[str, set] = {}
        for sentence in sentences:
            words = self._tokenize(sentence)
            for i, w1 in enumerate(words):
                if w1 not in word_connections:
                    word_connections[w1] = set()
                for w2 in words[i + 1:]:
                    word_connections[w1].add(w2)
        if not word_connections:
            return 0.0
        total = sum(len(v) for v in word_connections.values())
        avg = total / len(word_connections)
        return min(avg / 10.0, 1.0)

    def _compute_semantic_coherence(self, text: str) -> float:
        return 0.5 * self._compute_cohesion(text) + 0.5 * self._compute_lexical_overlap(text)

    def _compute_lexicon_frequency(self, text: str, lexicon: set) -> float:
        """
        Compute the proportion of words in `text` that belong to `lexicon`.
        Generic helper used by all new lexicon-based features.
        """
        words = self._tokenize(text)
        if not words:
            return 0.0
        return sum(1 for w in words if w in lexicon) / len(words)

    def _compute_punctuation_sentence_ratio(self, text: str, punctuation: str) -> float:
        """
        Compute the proportion of sentences that end with `punctuation`
        ('?' for questions, '!' for exclamations).
        """
        if not text.strip():
            return 0.0
        # Split keeping the delimiter so we can inspect it
        raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        if not raw_sentences:
            return 0.0
        count = sum(1 for s in raw_sentences if s.rstrip().endswith(punctuation))
        return count / len(raw_sentences)

    def _compute_incomplete_sentence_ratio(self, text: str, min_words: int = 3) -> float:
        """
        Compute the proportion of sentences with fewer than `min_words` words.
        High values may indicate thought disorganization (schizophrenia, ADHD).
        """
        sentences = self._get_sentences(text)
        if not sentences:
            return 0.0
        incomplete = sum(1 for s in sentences if len(self._tokenize(s)) < min_words)
        return incomplete / len(sentences)

    def _compute_mean_dependency_distance(self, text: str) -> float:
        """
        Compute the mean dependency distance (MDD) across all tokens using spaCy.
        MDD = average of |token.i - token.head.i| for non-root tokens.
        Higher MDD indicates more syntactically complex, harder-to-process sentences,
        which has been associated with anxiety and cognitive load.
        """
        doc = nlp(text)
        distances = [
            abs(token.i - token.head.i)
            for token in doc
            if token.dep_ != "ROOT" and token.is_alpha
        ]
        return round(float(np.mean(distances)), 4) if distances else 0.0

    def _compute_past_future_ratio(self, tense_dist: Dict[str, float]) -> float:
        """
        Compute ratio of past tense to future tense usage.
        past / (future + epsilon)
        High values → rumination / depression.
        Low values  → future-oriented / anxiety.
        Uses already-computed verb_tense_distribution to avoid double spaCy parsing.
        """
        past   = tense_dist.get("past",   0.0)
        future = tense_dist.get("future", 0.0)
        return round(float(np.log(past + 1) - np.log(future + 1)), 4)

    def _compute_repetition_rate(self, text: str) -> float:
        """
        Compute word repetition rate: proportion of words that are immediate
        repetitions of the previous word OR appear more than once in the text.
        Relevant for OCD (rumination) and disorganized thought (schizophrenia).
        Formula: (total_words - unique_words) / total_words  →  inverse of TTR.
        Unlike raw TTR this is framed as a *repetition* measure (higher = more repetition).
        """
        words = self._tokenize(text)
        if not words:
            return 0.0
        unique = len(set(words))
        return round(1.0 - unique / len(words), 4)

    def extract_temporal_features(self, texts: List[str], window_size: int = 5) -> np.ndarray:
        temporal_features = []
        for i in range(0, len(texts) - window_size + 1):
            window_texts = texts[i:i + window_size]
            window_markers = [self.extract_markers(t) for t in window_texts]
            temporal_features.append(self._compute_window_statistics(window_markers))
        return np.array(temporal_features)

    def _compute_window_statistics(self, markers_list: List[LinguisticMarkers]) -> np.ndarray:
        lexical_diversity  = [m.lexical_diversity  for m in markers_list]
        sentiment_polarity = [m.sentiment_polarity for m in markers_list]
        negation_freq      = [m.negation_frequency for m in markers_list]
        features = []
        for vals in [lexical_diversity, sentiment_polarity, negation_freq]:
            features.extend([np.mean(vals), np.std(vals), np.max(vals) - np.min(vals)])
        return np.array(features)

    def _compute_pos_frequencies(self, text: str) -> Dict[str, float]:
        """
        Compute POS-based frequency ratios using spaCy.
        - prep    (ADP):          prepositions — linked to concreteness/spatial thinking
        - auxverb (AUX):          auxiliary verbs — modal uncertainty, tense framing
        - adverb  (ADV):          adverbs — intensity, hedging
        - conj    (CCONJ/SCONJ):  conjunctions — discourse complexity
        """
        doc = nlp(text)
        tokens = [t for t in doc if t.is_alpha]
        if not tokens:
            return {"prep": 0.0, "auxverb": 0.0, "adverb": 0.0, "conj": 0.0}
        total = len(tokens)
        return {
            "prep":    round(sum(1 for t in tokens if t.pos_ == "ADP")              / total, 4),
            "auxverb": round(sum(1 for t in tokens if t.pos_ == "AUX")              / total, 4),
            "adverb":  round(sum(1 for t in tokens if t.pos_ == "ADV")              / total, 4),
            "conj":    round(sum(1 for t in tokens if t.pos_ in ("CCONJ", "SCONJ")) / total, 4),
        }

    def _compute_extended_pos(self, text: str) -> Dict[str, float]:
        """
        Extended POS frequencies: noun, verb, adjective, interjection.
        """
        doc = nlp(text)
        tokens = [t for t in doc if t.is_alpha]
        if not tokens:
            return {"noun": 0.0, "verb": 0.0, "adjective": 0.0, "interjection": 0.0}
        total = len(tokens)
        return {
            "noun":         round(sum(1 for t in tokens if t.pos_ == "NOUN")  / total, 4),
            "verb":         round(sum(1 for t in tokens if t.pos_ == "VERB")  / total, 4),
            "adjective":    round(sum(1 for t in tokens if t.pos_ == "ADJ")   / total, 4),
            "interjection": round(sum(1 for t in tokens if t.pos_ == "INTJ")  / total, 4),
        }

    def _compute_morphological_features(self, text: str) -> Dict[str, float]:
        """
        Morphological features from spaCy token.morph:
        - indicative_ratio: proportion of verbs in indicative mood (certainty)
        - subjunctive_ratio: proportion of verbs in subjunctive/conditional mood (hypotheticality)
        - singular_ratio: proportion of singular nouns (isolation vs collectivity)
        - plural_ratio: proportion of plural nouns
        """
        doc = nlp(text)
        verbs = [t for t in doc if t.pos_ in ("VERB", "AUX")]
        nouns = [t for t in doc if t.pos_ in ("NOUN", "PROPN")]

        result = {"indicative_ratio": 0.0, "subjunctive_ratio": 0.0,
                  "singular_ratio": 0.0, "plural_ratio": 0.0}

        if verbs:
            total_v = len(verbs)
            result["indicative_ratio"]  = round(sum(1 for t in verbs if "Ind" in t.morph.get("Mood")) / total_v, 4)
            result["subjunctive_ratio"] = round(sum(1 for t in verbs if t.morph.get("Mood") and
                                                    t.morph.get("Mood")[0] in ("Sub", "Cnd")) / total_v, 4)

        if nouns:
            total_n = len(nouns)
            result["singular_ratio"] = round(sum(1 for t in nouns if "Sing" in t.morph.get("Number")) / total_n, 4)
            result["plural_ratio"]   = round(sum(1 for t in nouns if "Plur" in t.morph.get("Number")) / total_n, 4)

        return result

    def _compute_dependency_features(self, text: str) -> Dict[str, float]:
        """
        Dependency tree features per sentence (averaged across sentences):
        - nsubj_rate: subjects per sentence (agentivity)
        - dobj_rate:  direct objects per sentence (thought transitivity)
        """
        doc = nlp(text)
        sentences = list(doc.sents)
        if not sentences:
            return {"nsubj_rate": 0.0, "dobj_rate": 0.0}

        nsubj_counts = [sum(1 for t in sent if t.dep_ == "nsubj") for sent in sentences]
        dobj_counts  = [sum(1 for t in sent if t.dep_ in ("dobj", "obj")) for sent in sentences]

        return {
            "nsubj_rate": round(float(np.mean(nsubj_counts)), 4),
            "dobj_rate":  round(float(np.mean(dobj_counts)),  4),
        }

    def _compute_ner_features(self, text: str) -> Dict[str, float]:
        """
        Named Entity Recognition features (normalized over total tokens):
        - person_ref_rate:    references to people (social isolation proxy)
        - temporal_ref_rate:  DATE + TIME entities (temporal orientation)
        """
        doc = nlp(text)
        total_tokens = len([t for t in doc if t.is_alpha])
        if total_tokens == 0:
            return {"person_ref_rate": 0.0, "temporal_ref_rate": 0.0}

        person_count   = sum(1 for ent in doc.ents if ent.label_ == "PERSON")
        temporal_count = sum(1 for ent in doc.ents if ent.label_ in ("DATE", "TIME"))

        return {
            "person_ref_rate":   round(person_count   / total_tokens, 4),
            "temporal_ref_rate": round(temporal_count / total_tokens, 4),
        }
    





def get_test_pvalues(group1_counts, group2_counts, alternative='two-sided'):
    """
    Args:
        group1_counts (list/array): Conteggi del primo gruppo
        group2_counts (list/array): Conteggi del secondo gruppo
        alternative (str): 'greater', 'less' o 'two-sided'
    """
    # T-test 
    _, p_value_ttest = stats.ttest_ind(group1_counts, group2_counts, alternative=alternative)
    
    # Mann-Whitney U
    _, p_value_mw = stats.mannwhitneyu(group1_counts, group2_counts, alternative=alternative)
    
    return p_value_ttest, p_value_mw

def create_clean_boxplot_save(data1, data2, filename, labels=None):
    if labels is None:
        labels = ['Depressed', 'Controls']
    
    plt.figure(figsize=(4, 3))
    sns.set_style("white")
    
    ax = sns.boxplot(data=[data1, data2], width=0.6, fliersize=5, linewidth=0, palette=['#A8D5BA', '#6D8A9A'])
    sns.stripplot(data=[data1, data2], jitter=True, alpha=0.5, size=7, color='#555555')
    
    plt.xticks([0, 1], labels)
    plt.ylabel('')
    plt.title('')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(left=False, bottom=False)
    #ax.set(yticks=[])
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
