import tensorflow as tf 
from config_gpt2 import GPT2Config
from gpt2SentencePretrain import GPT2Model
config = GPT2Config.from_pretrained(pretrained_model_name_or_path)
model = GPT2Model(config)