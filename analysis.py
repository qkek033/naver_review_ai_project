# app.py

import torch
from transformers import PreTrainedTokenizerFast, BartForConditionalGeneration
from transformers import ElectraTokenizer, ElectraForSequenceClassification

import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ====================
# 1. 모델 로드 (요약 + 감성)
# ====================

@st.cache_resource
def load_models():
    kobart_tokenizer = PreTrainedTokenizerFast.from_pretrained("digit82/kobart-summarization")
    kobart_model = BartForConditionalGeneration.from_pretrained("digit82/kobart-summarization")

    koelectra_tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-discriminator")
    koelectra_model = ElectraForSequenceClassification.from_pretrained(
        "monologg/koelectra-base-discriminator", num_labels=2)
    koelectra_model.eval()
    
    return kobart_tokenizer, kobart_model, koelectra_tokenizer, koelectra_model

def kobart_summarize(text, tokenizer, model):
    input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
    summary_ids = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def predict_sentiment(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted = torch.argmax(logits, dim=1).item()
    return "긍정" if predicted == 1 else "부정"