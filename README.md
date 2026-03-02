# TONY
TONY (**TO**lkit for **N**LP in Ps**Y**cology) is a Python package for Natural Language Processing (NLP) applied to mental health contexts.

The package combines two complementary approaches. First, it employs traditional lexicon-based analyses to extract linguistic markers and compute standard metrics that identify patterns in text. Second, it leverages transformer-based analyses using deep learning models to provide advanced predictions on emotions, psychological states, and clinical traits.

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


The Hierarchical Taxonomy of Psychopathology (HiTOP) is a dimensional framework that organizes psychopathology into a hierarchy of empirically derived constructs, from broad spectra (e.g., Internalizing, Thought Disorder, Detachment) down to specific maladaptive traits such as *anxiousness*, *withdrawal*, *depressivity*, or *emotional lability*. Unlike categorical diagnostic systems (e.g., DSM-5), HiTOP treats psychopathological features as continuous dimensions, enabling a finer-grained, transdiagnostic characterization of psychological functioning. Detecting these traits directly from language is therefore a clinically meaningful task, as each trait represents a stable, observable dimension of personality dysfunction that can manifest in naturalistic text.

```python
from TONY.HiTOP import HiTOP_Predictor
text = 'Some days I keep living, even though I feel completely alone in the world'
hitop = HiTOP_Predictor(model_name='FritzStack/HiTOP-Llama-3.2-3B_4bit-merged')
hitop.predict_HiTOP(text)
# Output: Anhedonia, Withdrawal, Depressivity
```


