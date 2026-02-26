from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class HiTOP_Predictor:

    def __init__(self, model_name='FritzStack/HiTOP-Llama-3.2-3B_4bit-merged'):

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True
        )

    def predict_HiTOP(self, text, max_new_tokens=200):
        prompt = f"{text}" + "HiTOP Traits:"
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

        return generated_text