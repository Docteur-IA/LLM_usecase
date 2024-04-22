# Psychologist/therapist usecase.
This usecase is reflected in the psychologist.py python file. Due to the time constraint and lack of GPU, I opted to use a pretrained model, that is well documented is a scientific paper.

## Model
I used a fine-tuned model called FLAN-T5-XXL with 4 high-quality text (6 tasks in total) datasets for the mental health prediction scenario: Dreaddit, DepSeverity, SDCNL, and CCRS-Suicide (see the reference paper). 

The performance is really good.

Reference papers
- https://arxiv.org/pdf/2307.14385.pdf
- https://arxiv.org/pdf/2307.11991.pdf

## Install requirements.
To run the code `python psychologist.py` please install the following :
```
!pip install transformers
!pip install llama-index llama-index-llms-huggingface
!pip install torch
!pip install bitsandbytes
!pip install git+https://github.com/huggingface/accelerate.git
```

# Summarization
Here I adopted to use a model which is based on the Facebook BART (Bidirectional and Auto-Regressive Transformers) architecture, specifically the large variant fine-tuned for text summarization tasks. BART is a sequence-to-sequence model introduced by Facebook AI, capable of handling various natural language processing tasks, including summarization.

# RAG

```
!pip install -q langchain==0.1.7
!pip install -q torch
!pip install -q transformers
!pip install -q sentence-transformers
!pip install -q datasets
!pip install -q faiss-cpu

```
## Dataset
The databricks-dolly-15k was selected as a dataset.





