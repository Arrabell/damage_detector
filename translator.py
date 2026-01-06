from transformers import pipeline
import torch
model = pipeline("translation",
                    model="Helsinki-NLP/opus-mt-en-uk",
                    torch_dtype=torch.float16,
                    device=0 if torch.cuda.is_available() else -1,
                    clean_up_tokenization_spaces=True, 
                    do_sample=False,
                    num_beams=4,
                    early_stopping=True,
                    no_repeat_ngram_size=2,
                    length_penalty=1.0, )
def predict_trans(text):
    results = model(text)
    return results