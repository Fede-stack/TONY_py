from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


class IRFPredictor:

    def __init__(self, model_name: str = "FritzStack/IRF-Qwen_8B_4bit-merged_2epo-mlx-4Bit", max_new_tokens: int = 200):
        self.max_new_tokens = max_new_tokens
        self.model, self.tokenizer = load(model_name)

    def highlight_evidence_IRF(self, text: str, max_new_tokens: int = None) -> str:
        """
        Predict Interpersonal Risk Factors for a given text: (i) Thwarted Belongingness (TBE), and (ii) Perceived Burdensomeness (PBU)
        """
        prompt_1 = """Question 1: Is there evidence of Thwarted Belongingness?
Answer: """
        prompt_2 = """Question 2: Is there evidence of Perceived Burdensomeness?
Answer: """
        prompt = text + "\n" + prompt_1

        sampler = make_sampler(temp=0.0, top_k=10)

        generated_text = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens or self.max_new_tokens,
            sampler=sampler,
            verbose=False,
        )

        return prompt_1 + generated_text.strip()