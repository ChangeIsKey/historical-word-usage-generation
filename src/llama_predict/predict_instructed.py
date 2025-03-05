from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, AddedToken, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTTrainer
from peft import PeftModel
from pathlib import Path
import sys
import os
import json


LLMs_CACHE_DIR = os.environ["TMPDIR"]
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
path_to_dataset = ""
base_model_name = ""
output_path = ""
temperature = 1.0

def formatting_func(batch_of_examples):
    batch_formatted = []
    for i in range(len(batch_of_examples['sampled_year'])):
        definition = batch_of_examples['definition'][i]
        year = batch_of_examples['sampled_year'][i]
        lemma = batch_of_examples['lemma'][i]
        prompt = f"""<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nRetrieve a sentence written in the year {year} in which the word {lemma} is used to mean {definition}. If you are not able to retrieve it, create a fictional sentence. The answer have to contain only the sentence.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"""
        batch_formatted.append(prompt)
    return batch_formatted


def load_data():
    test_dataset = load_dataset('json', data_files=path_to_dataset, split='train')
    return test_dataset


def eval():
    base_model_name_mod = base_model_name.split('/')[1]

    max_seq_length = 512

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side="left", cache_dir=LLMs_CACHE_DIR, trust_remote_code=True)

    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

    model = AutoModelForCausalLM.from_pretrained(
            f"{base_model_name}",
            device_map="auto",
            trust_remote_code=True,
            use_auth_token=True,
            quantization_config=bnb_config,
            cache_dir=LLMs_CACHE_DIR
    )

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    tokenizer.pad_token = tokenizer.eos_token

    test_dataset = load_data()
    model.eval()
    batch_size = 4
    n_sentences = 10

    with open(output_path,'w+') as f:
        with torch.no_grad():
            for ndx in range(0, len(test_dataset), batch_size):
                examples = {}
                
                examples['sampled_year'] = test_dataset['sampled_year'][ndx:min(ndx + batch_size, len(test_dataset))]
                examples['definition'] = test_dataset['definition'][ndx:min(ndx + batch_size, len(test_dataset))]
                examples['lemma'] = test_dataset['lemma'][ndx:min(ndx + batch_size, len(test_dataset))]
                examples['sense_id'] = test_dataset['sense_id'][ndx:min(ndx + batch_size, len(test_dataset))]
                
                model_input = tokenizer(formatting_func(examples), max_length=max_seq_length, truncation=True, padding=True, return_tensors="pt").to("cuda")
                output = model.generate(**model_input,
                                                temperature= temperature,
                                                repetition_penalty= 1.2,
                                                top_p = 0.9,
                                                do_sample=True,
                                                eos_token_id=terminators,
                                                num_return_sequences=n_sentences,
                                                max_new_tokens=256)
                input_tokens = model_input['input_ids'].size()[1]
                output = output[:,input_tokens:]
                results = tokenizer.batch_decode(output, skip_special_tokens=True)

                for j in range(0,len(results),n_sentences):
                    i = int(j / n_sentences)
                    D = {}
                    D['sampled_year'] = examples['sampled_year'][i]
                    D['definition'] = examples['definition'][i]
                    D['lemma'] = examples['lemma'][i]
                    D['sense_id'] = examples['sense_id'][i]
                    D['prediction'] = results[j:j+n_sentences]
                    f.write(json.dumps(D)+'\n')
                    f.flush()



eval()


