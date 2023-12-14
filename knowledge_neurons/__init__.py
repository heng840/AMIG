from transformers import BertTokenizer, BertLMHeadModel, GPT2Tokenizer, GPT2LMHeadModel, GPTNeoForCausalLM, \
    BartTokenizer, MBartForConditionalGeneration, BartForConditionalGeneration, AutoModel

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .knowledge_neurons import KnowledgeNeurons
from .garns import garns
from .data import pararel, pararel_expanded, PARAREL_RELATION_NAMES

BERT_MODELS = ["bert-base-uncased", "bert-base-multilingual-cased", "bert-base-cased"]
GPT2_MODELS = ["gpt2", "ai-forever/mGPT"]
GPT_NEO_MODELS = [
    "EleutherAI/gpt-neo-125M",
    "EleutherAI/gpt-neo-1.3B",
    "EleutherAI/gpt-neo-2.7B",
]
bart_models = ["facebook/mbart-large-50", 'facebook/bart-large']
ALL_MODELS = BERT_MODELS + GPT2_MODELS + GPT_NEO_MODELS + bart_models


def initialize_model_and_tokenizer(model_name: str):
    if model_name in BERT_MODELS:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertLMHeadModel.from_pretrained(model_name)
    elif model_name in GPT2_MODELS:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    elif model_name in GPT_NEO_MODELS:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPTNeoForCausalLM.from_pretrained(model_name)
    elif model_name in bart_models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        # model = AutoModel.from_pretrained(model_name)
    else:
        raise ValueError("Model {model_name} not supported")

    model.eval()

    return model, tokenizer


def model_type(model_name: str):
    if model_name in BERT_MODELS:
        return "bert"
    elif model_name in GPT2_MODELS:
        return "gpt"
    elif model_name in GPT_NEO_MODELS:
        return "gpt_neo"
    elif model_name in bart_models:
        return 'bart'
    else:
        raise ValueError("Model {model_name} not supported")