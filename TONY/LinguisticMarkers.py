import numpy as np
import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from collections import Counter
import string
import spacy
from collections import Counter
from nrclex import NRCLex
from typing import Dict
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('punkt_tab')
nltk.download('vader_lexicon', quiet=True)
import spacy
from typing import Dict
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

class LexiconLevelFeatures:
    """
    Class for extracting linguistic markers for various mental health conditions.

    Based on:
    - Natural Language Processing for mental health interventions
    - Multimodal approaches (text + optional acoustic features)
    """

    def __init__(self, language: str = "en"):
        """
        Initialize the marker extractor.

        Args:
            language: Language of the text to analyze (default: "en")
        """
        self.language = language
        self.condition_specific_markers = self._initialize_condition_markers()
        self._load_lexicons()

    def _load_lexicons(self):
        """Load linguistic lexicons and dictionaries for marker extraction"""
        # First person pronouns (increased in depression)
        self.first_person_pronouns = {
            'i', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves'
        }

        # Second person pronouns
        self.second_person_pronouns = {
            'you', 'your', 'yours', 'yourself', 'yourselves'
        }

        # Third person pronouns
        self.third_person_pronouns = {
            'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself',
            'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves'
        }

        # Negation words (increased in depression and anxiety)
        self.negation_words = {
            'not', 'no', 'never', 'nothing', 'none', 'nobody', 'nowhere',
            'neither', 'nor', "n't", 'cannot', 'cant', "won't", "don't",
            "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't"
        }

        # Absolutist words (increased in depression and anxiety)
        self.absolutist_words = {
            'always', 'never', 'everything', 'nothing', 'everyone', 'nobody',
            'all', 'none', 'every', 'completely', 'absolutely', 'totally',
            'entirely', 'forever', 'constant', 'permanent'
        }

        # Emotion lexicons (simplified LIWC-style)
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

        # Death and suicide related words (for suicide risk detection)
        self.death_words = {
            'death', 'die', 'dead', 'suicide', 'kill', 'end', 'gone',
            'disappear', 'cease', 'funeral', 'grave', 'coffin'
        }

        # Cognitive process words
        self.insight_words = {
            'think', 'know', 'understand', 'realize', 'believe', 'feel',
            'consider', 'recognize', 'wonder', 'imagine'
        }

        self.causation_words = {
            'because', 'cause', 'since', 'therefore', 'thus', 'hence',
            'consequently', 'as a result', 'due to', 'reason'
        }

        self.certainty_words = {
            'always', 'never', 'certainly', 'definitely', 'obviously',
            'clearly', 'undoubtedly', 'sure', 'certain'
        }

        self.tentative_words = {
            'maybe', 'perhaps', 'possibly', 'might', 'could', 'probably',
            'seem', 'appear', 'guess', 'suppose', 'uncertain'
        }

        # Social process words (reduced in depression)
        self.social_words = {
            'talk', 'share', 'friend', 'family', 'people', 'together',
            'social', 'community', 'relationship', 'connect', 'meet', 'girlfriend', 'boyfriend'
        }

        # Temporal markers
        self.past_tense_markers = {
            'was', 'were', 'had', 'did', 'been', 'ed'  # simplified
        }

        self.future_tense_markers = {
            'will', 'shall', 'going to', 'gonna', 'would', 'could',
            'might', 'may'
        }

        # Subordination markers
        self.subordinators = {
            'that', 'which', 'who', 'whom', 'whose', 'when', 'where',
            'if', 'because', 'although', 'though', 'while', 'since',
            'unless', 'until', 'before', 'after', 'as'
        }

        # Coordination markers
        self.coordinators = {
            'and', 'or', 'but', 'nor', 'yet', 'so', 'for'
        }

        # Connectives for cohesion
        self.connectives = {
            'however', 'therefore', 'moreover', 'furthermore', 'nevertheless',
            'consequently', 'additionally', 'meanwhile', 'thus', 'hence',
            'besides', 'otherwise', 'instead', 'then', 'next', 'finally'
        }

    def _initialize_condition_markers(self) -> Dict[MentalHealthCondition, List[str]]:
        """
        Define which markers are most relevant for each condition.
        Based on scientific literature.
        """
        return {
            MentalHealthCondition.DEPRESSION: [
                "lexical_diversity", "first_person_pronouns", "negative_emotions",
                "past_tense", "absolutist_words", "social_processes_low"
            ],
            MentalHealthCondition.ANXIETY: [
                "future_tense", "fear_words", "cognitive_processes_high",
                "tentative_language", "arousal_high", "sentence_complexity_high"
            ],
            MentalHealthCondition.BIPOLAR: [
                "lexical_variety_fluctuation", "positive_emotion_episodes",
                "activity_words", "flight_of_ideas", "grandiose_language"
            ],
            MentalHealthCondition.PTSD: [
                "trauma_related_words", "hypervigilance_markers",
                "avoidance_language", "emotional_numbing", "intrusive_thoughts"
            ],
            MentalHealthCondition.SCHIZOPHRENIA: [
                "graph_connectedness_low", "loosening_associations",
                "neologisms", "semantic_incoherence", "reduced_speech_connectivity"
            ],
            MentalHealthCondition.ADHD: [
                "stylistic_features", "lexical_richness", "cohesion_markers",
                "impulsive_language", "attention_shifts"
            ],
            MentalHealthCondition.STRESS: [
                "negative_affect", "worry_words", "time_pressure_words", "coping_language"
            ],
            MentalHealthCondition.SUICIDE_RISK: [
                "hopelessness_markers", "death_related_words", "isolation_language",
                "burden_perception", "lack_of_future_references"
            ],
        }

    def extract_markers(self, text: str,
                       condition: Optional[MentalHealthCondition] = None) -> LinguisticMarkers:
        """
        Extract linguistic markers from text.

        Args:
            text: Text to analyze
            condition: Specific condition to optimize extraction for

        Returns:
            LinguisticMarkers: Object containing all extracted markers
        """
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

        # Add specific markers for schizophrenia/psychosis
        if condition in [MentalHealthCondition.SCHIZOPHRENIA]:
            markers.graph_connectedness = self._compute_graph_connectedness(text)
            markers.semantic_coherence = self._compute_semantic_coherence(text)

        return markers

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        text = text.lower()
        text = re.sub(f'[{re.escape(string.punctuation)}]', ' ', text)
        words = text.split()

        return [w for w in words if w]

    def _lemmatize_words(self, text: str) -> List[str]:
        """Return list of lemmatized, lowercase, alphabetic tokens."""
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return [t.lemma_.lower() for t in doc if t.is_alpha]

    def _get_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting (could use nltk.sent_tokenize for better results)
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _compute_lexical_diversity(self, text: str) -> float:
        """
        Compute Type-Token Ratio (TTR) - measure of lexical diversity.
        Lower values indicate reduced vocabulary (common in depression).
        """
        words = self._tokenize(text)
        if len(words) == 0:
            return 0.0

        unique_words = len(set(words))
        total_words = len(words)

        # Type-Token Ratio
        ttr = unique_words / total_words

        return ttr

    def _compute_lexical_sophistication(self, text: str, reference_freq: dict = None) -> float:
        """
        Compute lexical sophistication using average inverse log frequency.
        Inspired by Kyle & Crossley (2015, TAALES framework).
        """
        words = self._tokenize(text)
        if not words:
            return 0.0

        # Se non viene fornito un dizionario di frequenze, usa frequenze di default
        if reference_freq is None:
            # Opzione 1: usa una frequenza uniforme
            freqs = [1e-3 for _ in words]
        else:
            # Opzione 2: usa il dizionario fornito
            freqs = [reference_freq.get(w, 1e-6) for w in words]

        inv_log_freqs = [-np.log10(f) for f in freqs]
        sophistication = float(np.mean(inv_log_freqs))
        return round(sophistication, 4)



    def _compute_word_prevalence(self, text: str) -> float:
        """
        Compute average word prevalence (how common words are).
        Lower values indicate use of less common words.
        """
        words = self._tokenize(text)
        if not words:
            return 0.0

        # Simplified: count words in our lexicons (more common words)
        common_word_count = sum(1 for word in words
                               if word in self.first_person_pronouns
                               or word in self.second_person_pronouns
                               or word in self.third_person_pronouns
                               or word in self.coordinators
                               or word in self.subordinators)

        prevalence = common_word_count / len(words)

        return prevalence

    def _compute_sentence_complexity(self, text: str) -> float:
        """
        Compute syntactic complexity based on sentence structure.
        Higher values indicate more complex sentences (common in anxiety).
        """
        sentences = self._get_sentences(text)
        if not sentences:
            return 0.0

        complexities = []
        for sentence in sentences:
            words = self._tokenize(sentence)
            if not words:
                continue

            # Count subordinators and coordinators
            subordination_count = sum(1 for w in words if w in self.subordinators)
            coordination_count = sum(1 for w in words if w in self.coordinators)

            # Complexity score based on clauses
            complexity = (subordination_count * 2 + coordination_count) / len(words)
            complexities.append(complexity)

        return np.mean(complexities) if complexities else 0.0

    def _compute_subordination_rate(self, text: str) -> float:
        """
        Compute rate of subordinate clauses.
        Indicates complex sentence structure.
        """
        words = self._tokenize(text)
        if not words:
            return 0.0

        subordination_count = sum(1 for w in words if w in self.subordinators)

        return subordination_count / len(words)

    def _compute_coordination_rate(self, text: str) -> float:
        """
        Compute rate of coordinate clauses.
        Indicates compound sentence structure.
        """
        words = self._tokenize(text)
        if not words:
            return 0.0

        coordination_count = sum(1 for w in words if w in self.coordinators)

        return coordination_count / len(words)

    def _extract_pronoun_usage(self, text: str) -> Dict[str, float]:
        """
        Extract pronoun usage distribution.
        Increased first-person pronoun use is associated with depression.
        """
        words = self._tokenize(text)
        if not words:
            return {"first_person": 0.0, "second_person": 0.0, "third_person": 0.0}

        total_words = len(words)

        first_person_count = sum(1 for w in words if w in self.first_person_pronouns)
        second_person_count = sum(1 for w in words if w in self.second_person_pronouns)
        third_person_count = sum(1 for w in words if w in self.third_person_pronouns)

        return {
            "first_person": first_person_count / total_words,
            "second_person": second_person_count / total_words,
            "third_person": third_person_count / total_words
        }



    def _extract_verb_tense(self, text: str) -> Dict[str, float]:
        """
        Extract verb tense distribution using spaCy morphological features.
        Based on the presence of Tense=Past/Pres/Fut in verbs.
        """
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)

        tenses = Counter({"past": 0, "present": 0, "future": 0})

        for token in doc:
            if token.pos_ == "VERB" or token.pos_ == "AUX":
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
        """Calcola frequenza negazioni"""
        negation_words = ["not", "no", "never", "nothing", "none", "non"]
        words = text.lower().split()
        return sum(1 for w in words if w in negation_words) / len(words) if words else 0.0


    def _extract_emotions(self, text: str) -> Dict[str, float]:
        """
        Extract emotion scores using NRC Emotion Lexicon (Mohammad & Turney, 2013).
        Returns normalized emotion proportions.
        """
        if not text.strip():
            return {emo: 0.0 for emo in ["joy", "sadness", "anger", "fear", "disgust", "surprise"]}

        nrc = NRCLex(text)
        raw_scores = nrc.raw_emotion_scores


        target_emotions = ["joy", "sadness", "anger", "fear", "disgust", "surprise", 'anticipation', 'trust']#, 'positive', 'negative']
        filtered_scores = {emo: raw_scores.get(emo, 0) for emo in target_emotions}

        total = sum(filtered_scores.values())
        if total == 0:
            return {emo: 0.0 for emo in target_emotions}

        # Normalize
        normalized = {emo: round(val / total, 4) for emo, val in filtered_scores.items()}
        return normalized

    def _compute_sentiment_polarity(self, text: str) -> float:
        """
        Compute sentiment polarity (negative to positive).
        Range: -1 (very negative) to +1 (very positive).
        """
        words = self._tokenize(text)
        if not words:
            return 0.0

        if not text.strip():
            return {emo: 0.0 for emo in ['positive', 'negative']}

        nrc = NRCLex(text)
        raw_scores = nrc.raw_emotion_scores  # es: {'fear': 3, 'anger': 2, ...}
        target_emotions = ['positive', 'negative']
        filtered_scores = {emo: raw_scores.get(emo, 0) for emo in target_emotions}
        #print(filtered_scores.values())
        positive_count = list(filtered_scores.values()) [0]
        negative_count = list(filtered_scores.values()) [1]
        total_sentiment_words = positive_count + negative_count

        if total_sentiment_words == 0:
            return 0.0

        # Polarity score
        polarity = (positive_count - negative_count) / total_sentiment_words

        return polarity



    def _compute_sentiment_intensity(self, text: str) -> float:

        """
        Compute overall sentiment intensity using VADER (Hutto & Gilbert, 2014).
        Returns a compound score in range [-1, 1]:
            -1 = extremely negative, +1 = extremely positive, 0 = neutral.
        """
        vader_analyzer = SentimentIntensityAnalyzer()
        if not text or not text.strip():
            return 0.0

        scores = vader_analyzer.polarity_scores(text)
        compound = scores["compound"]

        # opzionale: normalizza su scala 0–1 se preferisci un output positivo
        # normalized = (compound + 1) / 2

        return round(compound, 4)


    def _compute_cohesion(self, text: str) -> float:
        """
        Compute text cohesion based on sentence connectivity.
        Lower values indicate "flight of ideas" (common in mania/bipolar).
        """
        sentences = self._get_sentences(text)
        if len(sentences) < 2:
            return 1.0

        cohesion_scores = []

        for i in range(len(sentences) - 1):
            words1 = set(self._tokenize(sentences[i]))
            words2 = set(self._tokenize(sentences[i + 1]))

            if not words1 or not words2:
                continue

            # Jaccard similarity between consecutive sentences
            intersection = len(words1 & words2)
            union = len(words1 | words2)

            if union > 0:
                similarity = intersection / union
                cohesion_scores.append(similarity)

        return np.mean(cohesion_scores) if cohesion_scores else 0.0
    

    def _compute_lexical_overlap(self, text: str) -> float:
        """
        Compute lexical overlap between sentences.
        Related to cohesion but focuses on word repetition.
        """
        sentences = self._get_sentences(text)
        if len(sentences) < 2:
            return 0.0

        overlaps = []

        for i in range(len(sentences) - 1):
            words1 = self._tokenize(sentences[i])
            words2 = self._tokenize(sentences[i + 1])

            if not words1 or not words2:
                continue

            # Count overlapping words
            common_words = len(set(words1) & set(words2))
            overlap_ratio = common_words / min(len(words1), len(words2))
            overlaps.append(overlap_ratio)

        return np.mean(overlaps) if overlaps else 0.0

    def _compute_connectives(self, text: str) -> float:
        """
        Compute usage of discourse connectives.
        Indicates text organization and planning.
        """
        words = self._tokenize(text)
        if not words:
            return 0.0

        connective_count = sum(1 for w in words if w in self.connectives)

        return connective_count / len(words)

    def _extract_affect_scores(self, text: str) -> Dict[str, float]:
        """
        Approximate affect scores using NRC Emotion Lexicon.
        Maps emotions to valence-arousal space roughly.
        """
        if not text.strip():
            return {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}

        nrc = NRCLex(text)
        emotions = nrc.raw_emotion_scores

        positive = emotions.get("positive", 0)
        negative = emotions.get("negative", 0)
        anger = emotions.get("anger", 0)
        fear = emotions.get("fear", 0)
        joy = emotions.get("joy", 0)

        total = sum(emotions.values()) or 1
        valence = (positive - negative) / total
        arousal = (anger + fear + joy) / total
        dominance = 1 - fear / total  # euristica

        return {
            "valence": round(valence, 4),
            "arousal": round(arousal, 4),
            "dominance": round(dominance, 4)
        }

    def _extract_cognitive_processes(self, text: str) -> Dict[str, float]:
        """
        Extract cognitive process markers (LIWC-inspired).
        High cognitive processing associated with anxiety.
        Lemmatized comparison for higher lexical coverage.
        """
        lemmas = self._lemmatize_words(text)
        if not lemmas:
            return {"insight": 0.0, "causation": 0.0, "certainty": 0.0, "tentative": 0.0}

        total_words = len(lemmas)


        insight_lemmas = getattr(self, "_insight_lemmas", None)
        if insight_lemmas is None:
            self._insight_lemmas = {w for w in self.insight_words}
            self._causation_lemmas = {w for w in self.causation_words}
            self._certainty_lemmas = {w for w in self.certainty_words}
            self._tentative_lemmas = {w for w in self.tentative_words}
            insight_lemmas = self._insight_lemmas

        insight_count = sum(1 for w in lemmas if w in self._insight_lemmas)
        causation_count = sum(1 for w in lemmas if w in self._causation_lemmas)
        certainty_count = sum(1 for w in lemmas if w in self._certainty_lemmas)
        tentative_count = sum(1 for w in lemmas if w in self._tentative_lemmas)

        return {
            "insight": insight_count / total_words,
            "causation": causation_count / total_words,
            "certainty": certainty_count / total_words,
            "tentative": tentative_count / total_words,
        }


    def _extract_social_processes(self, text: str) -> float:
        """
        Extract social process markers.
        Reduced in depression (social withdrawal).
        Lemmatized comparison for accuracy.
        """
        lemmas = self._lemmatize_words(text)
        if not lemmas:
            return 0.0


        social_lemmas = getattr(self, "_social_lemmas", None)
        if social_lemmas is None:
            self._social_lemmas = {w for w in self.social_words}
            social_lemmas = self._social_lemmas

        social_count = sum(1 for w in lemmas if w in social_lemmas)
        return social_count / len(lemmas)


    def _compute_readability(self, text: str) -> float:
        """
        Compute Flesch Reading Ease score.
        Higher scores indicate easier-to-read text.
        """
        sentences = self._get_sentences(text)
        words = self._tokenize(text)

        if not sentences or not words:
            return 0.0

        # Count syllables (simplified: approximate by vowel groups)
        def count_syllables(word):
            vowels = 'aeiou'
            word = word.lower()
            syllable_count = 0
            previous_was_vowel = False

            for char in word:
                is_vowel = char in vowels
                if is_vowel and not previous_was_vowel:
                    syllable_count += 1
                previous_was_vowel = is_vowel

            # Adjust for silent e
            if word.endswith('e'):
                syllable_count -= 1

            # Ensure at least one syllable
            return max(1, syllable_count)

        total_syllables = sum(count_syllables(w) for w in words)

        # Flesch Reading Ease formula
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = total_syllables / len(words)

        flesch_score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word

        # Normalize to 0-1 scale
        normalized_score = max(0.0, min(flesch_score / 100.0, 1.0))

        return normalized_score

    def _compute_avg_sentence_length(self, text: str) -> float:
        """
        Compute average sentence length in words.
        Very long or very short sentences may indicate issues.
        """
        sentences = self._get_sentences(text)
        if not sentences:
            return 0.0

        sentence_lengths = [len(self._tokenize(s)) for s in sentences]

        return np.mean(sentence_lengths)

    def _compute_graph_connectedness(self, text: str) -> float:
        """
        Compute semantic graph connectedness.
        Lower values indicate "loosening of associations" (schizophrenia).
        Simplified: based on word co-occurrence patterns.
        """
        sentences = self._get_sentences(text)
        if len(sentences) < 2:
            return 1.0

        # Build simple word co-occurrence graph
        word_connections = {}

        for sentence in sentences:
            words = self._tokenize(sentence)
            for i, word1 in enumerate(words):
                if word1 not in word_connections:
                    word_connections[word1] = set()
                for word2 in words[i+1:]:
                    word_connections[word1].add(word2)

        # Calculate average connectivity
        if not word_connections:
            return 0.0

        total_connections = sum(len(connections) for connections in word_connections.values())
        avg_connections = total_connections / len(word_connections)

        # Normalize (assuming max 10 connections per word)
        connectedness = min(avg_connections / 10.0, 1.0)

        return connectedness

    def _compute_semantic_coherence(self, text: str) -> float:
        """
        Compute semantic coherence across sentences.
        Lower values indicate thought disorder (schizophrenia).
        Simplified: combination of cohesion and lexical overlap.
        """
        cohesion = self._compute_cohesion(text)
        overlap = self._compute_lexical_overlap(text)

        # Semantic coherence as weighted average
        coherence = 0.6 * cohesion + 0.4 * overlap

        return coherence

    def extract_temporal_features(self, texts: List[str],
                                  window_size: int = 5) -> np.ndarray:
        """
        Extract temporal features using sliding window.
        Useful for capturing longitudinal variations in language.

        Args:
            texts: List of texts ordered temporally
            window_size: Window size for calculation

        Returns:
            Numpy array with temporal distributions of features
        """
        temporal_features = []

        for i in range(0, len(texts) - window_size + 1):
            window_texts = texts[i:i + window_size]
            window_markers = [self.extract_markers(t) for t in window_texts]

            # Compute statistics on the window
            window_stats = self._compute_window_statistics(window_markers)
            temporal_features.append(window_stats)

        return np.array(temporal_features)

    def _compute_window_statistics(self, markers_list: List[LinguisticMarkers]) -> np.ndarray:
        """
        Compute statistics on temporal window of markers.
        Includes mean, std, trend for key features.
        """
        # Extract key features across window
        lexical_diversity = [m.lexical_diversity for m in markers_list]
        sentiment_polarity = [m.sentiment_polarity for m in markers_list]
        negation_freq = [m.negation_frequency for m in markers_list]

        # Compute statistics
        features = []
        for feature_values in [lexical_diversity, sentiment_polarity, negation_freq]:
            features.extend([
                np.mean(feature_values),
                np.std(feature_values),
                np.max(feature_values) - np.min(feature_values)  # range
            ])

        return np.array(features)
