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

```{python}
from TONY.Lexicon import MarkersExtraction, MarkersExtractionColab

app = MarkersExtractionColab() #Use MarkersExtraction if you are running it locally
```

<br><br>
<img src="https://raw.githubusercontent.com/Fede-stack/TONYpy/main/images/gif_extractmarker.gif" width="500">

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

The **BDI-II Scorer** module automatically completes the Beck Depression Inventory II (BDI-II) questionnaire from a user's post history, using an adaptive Retrieval-Augmented Generation (aRAG) pipeline. For each BDI-II item, the module dynamically retrieves the most relevant posts from the user's history and passes them to a generative LLM to produce a structured BDI-II response. Unlike standard RAG approaches that fix the number of retrieved documents a priori, the adaptive mechanism adjusts retrieval size based on the semantic density of the user's history relative to each item — retrieving more evidence when available, and less when the signal is sparse.

```python
from TONY.BDI import BDIScorer

posts = ['I have been feeling empty for weeks', 'I can barely get out of bed', ...]
scorer = BDIScorer(model_name='gemma-27B')
scorer.predict_BDI(posts)

# Output: 21-dimensional vector of predicted BDI-II item scores
```

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

Vuoi che uniformi lo stile di tutti e quattro i moduli (HiTOP, IRF, BDIScorer, SAEInterpreter) in un unico documento di documentazione?


