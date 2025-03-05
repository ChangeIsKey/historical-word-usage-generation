from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import sys
import os
import re
import json


LLMs_CACHE_DIR = os.environ["TMPDIR"]
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def formatting_func(batch_of_examples):
    batch_formatted = []
    for i in range(len(batch_of_examples['sampled_year'])):
        definition = batch_of_examples['definition'][i]
        year = batch_of_examples['sampled_year'][i]
        lemma = batch_of_examples['lemma'][i]
        text = f"{year}<|t|>{lemma}<|t|>{definition}<|s|>"
        batch_formatted.append(text)
    return batch_formatted




def eval():
    max_seq_length = 512

    tokenizer = AutoTokenizer.from_pretrained('ChangeIsKey/llama3-janus', cache_dir=LLMs_CACHE_DIR, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
            'ChangeIsKey/llama3-janus',
            device_map="auto",
            trust_remote_code=True,
            use_auth_token=True,
            cache_dir=LLMs_CACHE_DIR
    )


    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|end|>"),
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    tokenizer.pad_token = tokenizer.eos_token

    definitions = [
        'Of poor quality or little worth.',
        'Demonstrating or indicative of profound reverence or respect; done or performed respectfully or with great reverence.',
        'Used to emphasize something unpleasant or negative; ‘such a’, ‘an absolute’.',
        'Of a surface: Without curvature, indentation, or protuberance; plane, level.',
        'A well-known carnivorous quadruped (Felis domesticus) which has long been domesticated, being kept to destroy mice, and as a house pet.',
        'A mixture of dissimilar qualities or traits.',
        'A conveyance, a form of transport.',
        'The fact or condition of being, or of having been, written down as evidence of a legal matter, esp. as part of the proceedings or verdict of a court of law; evidence which is preserved in this way, and may be appealed to in case of dispute.',
        'The known history of the life or career of a person, esp. a public figure; the sum of the past activities, achievements, or performance of person, organization, etc.'
    ]

    lemmas = [
        'good',
        'awful',
        'awful',
        'tably',
        'chair',
        'zebra',
        'train',
        'record',
        'record'
    ]

    years = [2020,2020,1800,2020,2020,2020,2020,2020,1800]

    test_dataset = {
    'sampled_year': years,
    'definition': definitions,
    'lemma': lemmas,
    }
    
    model.eval()
    batch_size = 9
    n_sentences = 30

    with open(f'output_tokens.jsonl','w+') as f:
        with torch.no_grad():
            for ndx in range(0, len(test_dataset), batch_size):
                examples = {}
                
                examples['sampled_year'] = test_dataset['sampled_year'][ndx:min(ndx + batch_size, len(test_dataset['sampled_year']))]
                examples['definition'] = test_dataset['definition'][ndx:min(ndx + batch_size, len(test_dataset['sampled_year']))]
                examples['lemma'] = test_dataset['lemma'][ndx:min(ndx + batch_size, len(test_dataset['sampled_year']))]
                
                model_input = tokenizer(formatting_func(examples), max_length=max_seq_length, truncation=True, padding=True, return_tensors="pt").to("cuda")
                output = model.generate(**model_input,
                                                temperature=1.0,
                                                repetition_penalty= 1.2,
                                                top_p = 0.9,
                                                do_sample=True,
                                                eos_token_id=terminators,
                                                num_return_sequences=n_sentences,
                                                max_new_tokens=256)
                input_tokens = model_input['input_ids'].size()[1]
                output = output[:,input_tokens:]
                results = tokenizer.batch_decode(output, skip_special_tokens=False)

                for j in range(0,len(results),n_sentences):
                    i = int(j / n_sentences)
                    D = {}
                    D['sampled_year'] = examples['sampled_year'][i]
                    D['definition'] = examples['definition'][i]
                    D['lemma'] = examples['lemma'][i]
                    D['prediction'] = results[j:j+n_sentences]
                    f.write(json.dumps(D)+'\n')
                    f.flush()


def extract_text(fname, fname_clean):
    errors = {}

    errors[fname] = 0
    with open(fname) as f:
        with open(fname_clean,'w+') as g:
            for line in f:
                line = json.loads(line)
                D = dict(line)
                D['prediction'] = []
                D['prediction_offsets'] = []
                for p in line['prediction']:
                    p = p.replace('<|end_of_text|>','')
                    p = p.replace('<|end|>','')
                    results = [r for r in re.compile(r'<\|t\|>').finditer(p)]
                    if not len(results) == 2:
                        errors[fname] = errors[fname] + 1
                    else:
                        start = results[0].span()[0]
                        end = results[1].span()[1]
                        new_string = p[:start] + p[start+5:end-5] + p[end:]
                        new_start = start
                        new_end = start + (end-start-10)
                        D['prediction'].append(new_string)
                        D['prediction_offsets'].append([new_start,new_end])
                g.write(json.dumps(D)+'\n')



eval()
extract_text('output_tokens.jsonl','texts.jsonl')


