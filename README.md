# TONY
TONY (**TO**lkit for **N**LP in Ps**Y**cology) is a Python package for Natural Language Processing (NLP) applied to mental health-related text data.

The package combines two complementary approaches. First, it employs traditional Linguistic-based analyses to extract linguistic markers and compute standard metrics that identify patterns in text. Second, it leverages transformer-based analyses using deep learning models to provide advanced predictions on emotions, psychological states, and clinical traits.

This combination allows researchers and practitioners to analyze mental health-related texts using both interpretable methods and state-of-the-art techniques, offering flexibility for research and clinical applications.

 <br><br>
<img src="https://github.com/Fede-stack/TONY_py/blob/main/images/overview.png" alt="" width="1000">

## Getting Started

Overview of how to begin using the package.

### Installation

```{python}
!pip install git+https://github.com/Fede-stack/TONYpy.git
```

### Quick Start

#### How to extract Linguistic Markers?

> [!NOTE]
> TONY uses spaCy under the hood. Before running the cell below, install the English language model:
> ```bash
> python -m spacy download en_core_web_sm
> ```

```{python}
from TONY.Lexicon import MarkersExtraction, MarkersExtractionColab

app = MarkersExtractionColab() #Use MarkersExtraction if you are running it locally
```

<br><br>
<img src="https://raw.githubusercontent.com/Fede-stack/TONYpy/main/images/gif_extractmarkers.gif" width="500">

Alternatively, you can extract features without using the ui with two lines of code: 

```{python}
text = 'Some days I keep living, even though I feel completely alone in the world'
markers = LexiconLevelFeatures(language="en")
markers.extract_markers(text)
# Output:
# LinguisticMarkers(lexical_diversity=0.875, lexical_sophistication={'mean_frequency': 0.002983, 'std_frequency': 0.004642}, word_prevalence=0.25, sentence_complexity=0.0, subordination_rate=0.0, coordination_rate=0.0, pronoun_usage={'first_person': 0.125, 'second_person': 0.125, 'third_person': 0.0}, verb_tense_distribution={'past': 0.0, 'present': 1.0, 'future': 0.0}, negation_frequency=0.0, emotion_scores={'joy': 0.0, 'sadness': 0.0, 'anger': 0.0, 'fear': 0.0, 'disgust': 0.0, 'surprise': 0.0, 'anticipation': 0.0, 'trust': 0.0}, sentiment_polarity=0.0, sentiment_intensity=0.4717, cohesion_score=1.0, lexical_overlap=0.0, connectives_usage=0.0, affect_scores={'valence': 0.0, 'arousal': 0.0, 'dominance': 1.0}, cognitive_processes={'insight': 0.0, 'causation': 0.0, 'certainty': 0.0, 'tentative': 0.0}, social_processes=0, readability_index=1.0, average_sentence_length=8.0, graph_connectedness=1.0, semantic_coherence=0.5, absolutist_word_frequency=0.0, death_word_frequency=0.0, anxiety_word_frequency=0.0, sadness_word_frequency=0.0, anger_word_frequency=0.0, question_ratio=0.0, exclamation_ratio=0.0, incomplete_sentence_ratio=0.0, mean_dependency_distance=2.1667, past_future_ratio=0.0, repetition_rate=0.125, body_word_frequency=0.0, achievement_word_frequency=0.0, pos_frequencies={'prep': 0.1429, 'auxverb': 0.1429, 'adverb': 0.0, 'conj': 0.0}, extended_pos={'noun': 0.1429, 'verb': 0.2857, 'adjective': 0.0, 'interjection': 0.0}, morphological_features={'indicative_ratio': 0.3333, 'subjunctive_ratio': 0.0, 'singular_ratio': 1.0, 'plural_ratio': 0.0}, dependency_features={'nsubj_rate': 2.0, 'dobj_rate': 1.0}, ner_features={'person_ref_rate': 0.0, 'temporal_ref_rate': 0.0})
```

#### LLMs Features


The Hierarchical Taxonomy of Psychopathology (**HiTOP**) is a dimensional framework that organizes psychopathology into a hierarchy of empirically derived constructs, from broad spectra (e.g., Internalizing, Thought Disorder, Detachment) down to specific maladaptive traits such as *anxiousness*, *withdrawal*, *depressivity*, or *emotional lability*. Unlike categorical diagnostic systems (e.g., DSM-5), HiTOP treats psychopathological features as continuous dimensions, enabling a finer-grained, transdiagnostic characterization of psychological functioning. Detecting these traits directly from language is therefore a clinically meaningful task, as each trait represents a stable, observable dimension of personality dysfunction that can manifest in naturalistic text.
With **TONY** you can extract HiTOP traits from text with just a few lines of code, leveraging a lightweight fine-tuned LLM that runs efficiently on consumer hardware. If you are working with an Apple Silicon Mac (M1/M2/M3/M4 chip), you can choose to run the model locally using **MLX**, Apple's machine learning framework optimized for the Metal Performance Shaders backend, enabling fast and energy-efficient inference directly on your device — no GPU server or internet connection required.

```python
from TONY.HiTOP import HiTOP_Predictor, HiTOP_Predictor_mlx
text = 'Some days I keep living, even though I feel completely alone in the world'
hitop = HiTOP_Predictor(model_name='FritzStack/HiTOP-Llama-3.2-3B_4bit-merged')
hitop.predict_HiTOP(text)
# Output: Anhedonia, Withdrawal, Depressivity
```

---

The **Interpersonal Risk Factors** (**IRF**) module detects the two core interpersonal risk factors defined by the Interpersonal Theory of Suicide (Van Orden et al., 2010): **Thwarted Belongingness** (TBE) — the painful feeling of being disconnected from others — and **Perceived Burdensomeness** (PBU) - the perception of being a liability to those around oneself. The module not only predicts the presence of each factor but also highlights the supporting textual evidence, providing interpretable outputs for both clinical and research use.

```python
from TONY.IRF import IRFPredictor, IRFPredictor_mlx

text = 'Some days I keep living, even though I feel completely alone in the world'
irf = IRFPredictor(model_name='FritzStack/IRF-Qwen3-8B_4bit-merged')
irf.highlight_evidence_IRF(text)

# Question 1: Is there evidence of Thwarted Belongingness?
# Answer: Yes
# Text Evidence: feel completely alone

# Question 2: Is there evidence of Perceived Burdensomeness?
# Answer: No
# Text Evidence: nan
```

If you are working with an Apple Silicon Mac (M1/M2/M3/M4 chip), you can run the model locally using **MLX**, enabling fast and energy-efficient inference without requiring a GPU or internet connection.

---

The **SAE Interpreter** module provides interpretable latent feature analysis using a Sparse Autoencoder (SAE) trained on 710,000 Reddit posts spanning from casual conversation to mental health-focused communities. Given an input text, the model identifies the most strongly activated latent features, each of which is automatically described in natural language, capturing the psychological and semantic content expressed in the text.

```python
from TONY.SAE import SAEInterpreter

text = 'Some days I keep living, even though I feel completely alone in the world'
interpreter = SAEInterpreter()
result = interpreter.interpret(text, top_k=10)
interpreter.plot_interpretation(result)

# Feature #25 activated
# This feature captures posts that involve questioning or describing
# unusual perceptual experiences, often related to dissociation or
# altered states of consciousness.
```

---

Additionally, you can visualize the documents into clusters of topics. Following an example with a 5000 sample coming from Reddit Mental Health dataset


```python
from TONY.viz import visualize_topics

clusterer = visualize_topics(n_clusters=5)
clusterer.fit(list(texts))
clusterer.plot()
```

 <br><br>
<img src="https://github.com/Fede-stack/TONY_py/blob/main/images/cluster_viz.png" alt="" width="400">


---

## Filling BDI-II Questionnaire

The **BDI-II Scorer** module automatically completes the Beck Depression Inventory II (BDI-II) questionnaire from a user's post history, using the adaptive Retrieval-Augmented Generation (aRAG) pipeline by Ravenda et al. For each BDI-II item, the module dynamically retrieves the most relevant posts from the user's history and passes them to a generative LLM to produce a structured BDI-II response. Unlike standard RAG approaches that fix the number of retrieved documents a priori, the adaptive mechanism adjusts retrieval size based on the semantic density of the user's history relative to each item, retrieving more evidence when available, and less when the signal is sparse.

```python
from TONY.BDI import BDIScorer

bdi_items = [
    # 1. Sadness
    [
        "I do not feel sad.",
        "I feel sad much of the time.",
        "I am sad all the time.",
        "I am so sad or unhappy that I can't stand it."
    ],
    # 2. Pessimism
    [
        "I am not discouraged about my future.",
        "I feel more discouraged about my future than I used to be.",
        "I do not expect things to work out for me.",
        "I feel my future is hopeless and will only get worse."
    ],
    # 3. Past Failure
    [
        "I do not feel like a failure.",
        "I have failed more than I should have.",
        "As I look back, I see a lot of failures.",
        "I feel I am a total failure as a person."
    ],
    # 4. Loss of Pleasure
    [
        "I get as much pleasure as I ever did from the things I enjoy.",
        "I don't enjoy things as much as I used to.",
        "I get very little pleasure from the things I used to enjoy.",
        "I can't get any pleasure from the things I used to enjoy."
    ],
    # 5. Guilty Feelings
    [
        "I don't feel particularly guilty.",
        "I feel guilty over many things I have done or should have done.",
        "I feel quite guilty most of the time.",
        "I feel guilty all of the time."
    ],
    # 6. Punishment Feelings
    [
        "I don't feel I am being punished.",
        "I feel I may be punished.",
        "I expect to be punished.",
        "I feel I am being punished."
    ],
    # 7. Self-Dislike
    [
        "I feel the same about myself as ever.",
        "I have lost confidence in myself.",
        "I am disappointed in myself.",
        "I dislike myself."
    ],
    # 8. Self-Criticalness
    [
        "I don't criticize or blame myself more than usual.",
        "I am more critical of myself than I used to be.",
        "I criticize myself for all of my faults.",
        "I blame myself for everything bad that happens."
    ],
    # 9. Suicidal Thoughts or Wishes
    [
        "I don't have any thoughts of killing myself.",
        "I have thoughts of killing myself, but I would not carry them out.",
        "I would like to kill myself.",
        "I would kill myself if I had the chance."
    ],
    # 10. Crying
    [
        "I don't cry anymore than I used to.",
        "I cry more than I used to.",
        "I cry over every little thing.",
        "I feel like crying, but I can't."
    ],
    # 11. Agitation
    [
        "I am no more restless or wound up than usual.",
        "I feel more restless or wound up than usual.",
        "I am so restless or agitated that it's hard to stay still.",
        "I am so restless or agitated that I have to keep moving or doing something."
    ],
    # 12. Loss of Interest
    [
        "I have not lost interest in other people or activities.",
        "I am less interested in other people or things than before.",
        "I have lost most of my interest in other people or things.",
        "It's hard to get interested in anything."
    ],
    # 13. Indecisiveness
    [
        "I make decisions about as well as ever.",
        "I find it more difficult to make decisions than usual.",
        "I have much greater difficulty in making decisions than I used to.",
        "I have trouble making any decisions."
    ],
    # 14. Worthlessness
    [
        "I do not feel I am worthless.",
        "I don't consider myself as worthwhile and useful as I used to.",
        "I feel more worthless as compared to other people.",
        "I feel utterly worthless."
    ],
    # 15. Loss of Energy
    [
        "I have as much energy as ever.",
        "I have less energy than I used to have.",
        "I don't have enough energy to do very much.",
        "I don't have enough energy to do anything."
    ],
    # 16. Changes in Sleeping Pattern
    [
        "I have not experienced any change in my sleeping pattern.",
        "I sleep somewhat more than usual OR I sleep somewhat less than usual.",
        "I sleep a lot more than usual OR I sleep a lot less than usual.",
        "I sleep most of the day OR I wake up 1-2 hours early and can't get back to sleep."
    ],
    # 17. Irritability
    [
        "I am no more irritable than usual.",
        "I am more irritable than usual.",
        "I am much more irritable than usual.",
        "I am irritable all the time."
    ],
    # 18. Changes in Appetite
    [
        "I have not experienced any change in my appetite.",
        "My appetite is somewhat less than usual OR My appetite is somewhat greater than usual.",
        "My appetite is much less than before OR My appetite is much greater than usual.",
        "I have no appetite at all OR I crave food all the time."
    ],
    # 19. Concentration Difficulty
    [
        "I can concentrate as well as ever.",
        "I can't concentrate as well as usual.",
        "It's hard to keep my mind on anything for very long.",
        "I find I can't concentrate on anything."
    ],
    # 20. Tiredness or Fatigue
    [
        "I am no more tired or fatigued than usual.",
        "I get more tired or fatigued more easily than usual.",
        "I am too tired or fatigued to do a lot of the things I used to do.",
        "I am too tired or fatigued to do most of the things I used to do."
    ],
    # 21. Loss of Interest in Sex
    [
        "I have not noticed any recent change in my interest in sex.",
        "I am less interested in sex than I used to be.",
        "I am much less interested in sex now.",
        "I have lost interest in sex completely."
    ]
]

items_names = ['Sadness', 'Pessimism', 'Past Failure', 'Loss of Pleasure', 'Guilty Feelings', 'Punishment Feelings', 
              'Self-Dislike', 'Self-Criticalness', 'Suicidal Thoughts or Wishes', 'Crying', 'Agitation', 
              'Loss of Interest', 'Indecisiveness', 'Worthlessness', 'Loss of Energy', 'Changes in Sleeping Pattern', 
              'Irritability', 'Changes in Appetite', 'Concentration Difficulty', 'Tiredness or Fatigue', 
              'Loss of Interest in Sex']


posts = [['I have been feeling empty for weeks', 'I can barely get out of bed', ...]]  # Each inner list contains all Reddit posts written by a single user
scorer = BDIScorer(
    retriever_model_name='FritzStack/mpnet_MH_embedding',
    llm_model_name='google/gemma-3-27b-it',
    use_hf=False,
    client=client,
)
response_llms = scorer.score(reddit_posts, bdi_items, items_names)

# Output: 21-dimensional vector of predicted BDI-II item scores
```


