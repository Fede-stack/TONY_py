from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class IRFPredictor:

    def __init__(self, model_name='FritzStack/IRF-Qwen_8B_4bit-merged_2epo'):
        self.prompt_1 = """Question 1: Is there evidence of Thwarted Belongingness?
Answer: """
        self.prompt_2 = """Question 2: Is there evidence of Perceived Burdensomeness?
Answer: """

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

    def predict_irf(self, text, max_new_tokens=200):
        prompt = f"{text}\n" + self.prompt_1
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_k=10,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_text = self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        ).strip()

        return self.prompt_1 + generated_text
