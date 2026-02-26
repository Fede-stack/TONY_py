from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


class HiTOPPredictor:

    def __init__(self, model_name: str = "FritzStack/HiTOP-Llama-3.2-3B_4bit-merged-mlx-4Bit", max_new_tokens: int = 200):
        self.max_new_tokens = max_new_tokens
        self.model, self.tokenizer = load(model_name)

    def predict_HiTOP(self, text: str, max_new_tokens: int = None) -> str:
        """
        Predict HiTOP traits for a given text.
        """
        prompt = text + "HiTOP Traits:" 

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