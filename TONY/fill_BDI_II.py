import re
import random
import itertools
import warnings
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings('ignore')


class BDIScorer:

    def __init__(
        self,
        retriever_model_name='FritzStack/mpnet_MH_embedding',
        llm_model_name='openai/gpt-4o-mini',
        use_hf=False,
        client=None,
        k_fallback=5,
        random_seed=0,
    ):
        self.llm_model_name = llm_model_name
        self.use_hf = use_hf
        self.client = client
        self.random_seed = random_seed

        # Retriever
        from .adaptRetriever import aRAG  # o il tuo import
        self.retriever = aRAG(
            model_name=retriever_model_name,
            k_fallback=k_fallback,
            random_seed=random_seed
        )

        # HuggingFace LLM (lazy load solo se use_hf=True)
        if use_hf:
            self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                torch_dtype=torch.float16,
                device_map='auto',
                trust_remote_code=True
            )
        else:
            self.tokenizer = None
            self.model = None

    def _set_seeds(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)

    def _build_prompt(self, documents_retrieved, item_name, content):
        return f"""
ROLE:
You are a psychological assessment assistant.

TASK:
Estimate the most appropriate response to a specific BDI-II item,
based only on the information contained in the retrieved Reddit posts.

CONTEXT:
Retrieved Reddit posts:
{documents_retrieved}

BDI-II Item:
{item_name}

Response options (score 0–3):
{content}

INSTRUCTIONS:
- Choose the single most appropriate score (0, 1, 2, or 3).
- Base your decision only on the retrieved posts.
- Do not infer information that is not explicitly supported.

OUTPUT FORMAT:
Return only one number: 0, 1, 2, or 3.
Do not provide explanations.
Answer:
"""

    def _query_hf(self, prompt, max_new_tokens=10):
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        ).strip()

    def _query_client(self, prompt):
        response = self.client.chat.completions.create(
            model=self.llm_model_name,
            messages=[{'role': 'user', 'content': prompt}],
            temperature=0,
            max_tokens=10
        )
        return response.choices[0].message.content

    def _parse_response(self, raw):
        if raw is None:
            return '0'
        match = re.search(r'[0-3]', raw.strip())
        return match.group(0) if match else '0'

    def score(self, docss, sentences_bdi, bdi_queries, bdi_items, items_names):
        """
        Parameters
        ----------
        docss        : list of users, each user is a list of posts
        sentences_bdi: list of sentences used for retrieval
        bdi_queries  : list of BDI query strings
        bdi_items    : list of lists of response options per item
        items_names  : list of BDI item names

        Returns
        -------
        response_llms : list of lists — one score per item per user
        """
        self._set_seeds()
        response_llms = []

        for j, user in enumerate(docss):
            response_user = []
            print('#' * 20)

            relevant_docs = self.retriever.retrieve_batch(sentences_bdi, user)

            for i, query in enumerate(bdi_queries):
                documents_retrieved = list(set(list(
                    itertools.chain.from_iterable([relevant_docs[i + k] for k in range(4)])
                )))
                content = ''.join([
                    str(idx) + ' ' + item + '\n '
                    for idx, item in enumerate(bdi_items[i])
                ])
                prompt = self._build_prompt(documents_retrieved, items_names[i], content)

                try:
                    if self.use_hf:
                        raw = self._query_hf(prompt)
                    else:
                        raw = self._query_client(prompt)
                    response_user.append(self._parse_response(raw))

                except Exception as e:
                    print(f'  [ERROR] user {j}, item {i}: {e} — defaulting to 0')
                    response_user.append('0')

            print(response_user)
            response_llms.append(response_user)

        return response_llms
