from transformers import AutoTokenizer, AutoModelForCausalLM

from llama_index.llms.huggingface import HuggingFaceLLM

import torch
from transformers import BitsAndBytesConfig
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.huggingface import HuggingFaceLLM


def messages_to_prompt(messages):
    '''
        Message to prompt function
    '''
    prompt = ""
    for message in messages:
        if message.role == 'system':
          prompt += f"<|system|>\n{message.content}</s>\n"
        elif message.role == 'user':
          prompt += f"<|user|>\n{message.content}</s>\n"
        elif message.role == 'assistant':
          prompt += f"<|assistant|>\n{message.content}</s>\n"

    # ensure we start with a system prompt, insert blank if needed
    if not prompt.startswith("<|system|>\n"):
        prompt = "<|system|>\n</s>\n" + prompt

    # add final assistant prompt
    prompt = prompt + "<|assistant|>\n"

    return prompt

def completion_to_prompt(completion):
    '''
        Format of completing the prompt
    '''
    return f"<|system|>\n</s>\n<|user|>\n{completion}</s>\n<|assistant|>\n"




# quantize to save memory
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
# the pretrained LLM 
llm = HuggingFaceLLM(
    model_name="NEU-HAI/mental-flan-t5-xxl",
    tokenizer_name="NEU-HAI/mental-flan-t5-xxl",
    context_window=3900,
    max_new_tokens=256,
    model_kwargs={"quantization_config": quantization_config},
    generate_kwargs={"temperature": 0.7, "top_k": 50, "top_p": 0.95},
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=completion_to_prompt,
    device_map="auto",
)


response = llm.complete("I am very angry ")
print(str(response))