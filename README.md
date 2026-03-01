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
pip install !pip install git+https://github.com/Fede-stack/TONYpy.git
```

### Quick Start

#### Linguistic Markers

```{python}
from TONY.Lexicon import MarkersExtraction, MarkersExtractionColab

app = MarkersExtractionColab() #Use MarkersExtraction if you are running it locally
```

<br><br>
<img src="https://raw.githubusercontent.com/Fede-stack/TONYpy/main/images/gif_extractmarker.gif" width="500">
