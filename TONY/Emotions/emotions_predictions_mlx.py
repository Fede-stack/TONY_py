from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


class Emotions_Predictor_mlx:

    def __init__(self, model_name: str = "FritzStack/QWEN4B-GoEmotions-mlx-Q4", max_new_tokens: int = 200):
        self.max_new_tokens = max_new_tokens
        self.model, self.tokenizer = load(model_name)

    def predict_emotions(self, text: str, max_new_tokens: int = None) -> str:
        """
        Predict Emotions for a given text.
        """
        prompt = text + "\n Emotions Output:" 

        sampler = make_sampler(temp=0.0, top_k=10)

        generated_text = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens or self.max_new_tokens,
            sampler=sampler,
            verbose=False,
        )

        return generated_text.strip()
