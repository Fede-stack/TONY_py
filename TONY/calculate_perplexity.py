from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import math


class PerplexityExtractor:
    """
    Class to extract perplexity from a causal language model.
    
    Parameters:
    -----------
    model_name : str
        The name or path of the pre-trained model from Hugging Face
    device : str, optional
        Device to run the model on ('cuda' or 'cpu'). Auto-detects if None
    """
    
    def __init__(self, model_name, device=None):
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Set pad token if not available
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def compute_perplexity(self, text):
        """
        Compute perplexity for a given text.
        
        Parameters:
        -----------
        text : str
            The input text to compute perplexity for
            
        Returns:
        --------
        float
            The perplexity score
        """
        # Tokenize input text
        inputs = self.tokenizer(str(text), return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Compute loss with no gradient tracking
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
        
        # Calculate perplexity from cross-entropy loss
        perplexity = math.exp(loss.item())
        
        return perplexity
    
    def compute_perplexity_batch(self, texts, batch_size=8):
        """
        Compute perplexity for multiple texts.
        
        Parameters:
        -----------
        texts : list of str
            List of input texts
        batch_size : int
            Batch size for processing
            
        Returns:
        --------
        list of float
            List of perplexity scores for each text
        """
        perplexities = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            for text in batch:
                ppl = self.compute_perplexity(text)
                perplexities.append(ppl)
        
        return perplexities
