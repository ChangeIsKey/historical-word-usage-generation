import openai
import json
import re

OPENAI_API_KEY = ''
openai.api_key = OPENAI_API_KEY

model = "gpt-4o"
path_to_dataset = ""
output_path = ""


example_prompts = [
    {'role':"user", 'content':"""Retrieve a sentence written in the year 1841 in which the word fourth is used to mean "A player who comes in to complete a party of four in a game or at a social event". If you are not able to retrieve it, create a fictional sentence. The answer have to contain only the sentence."""},
    {'role':"assistant", 'content':"Browne dines with me. The gay and festive Thompson also joins us. Will you make a fourth?"},
    {'role':"user", 'content':"""Retrieve a sentence written in the year 1721 in which the word matronship is used to mean "The character of a matron. Chiefly humorously with possessive adjective, as a title of mock respect given to a matron, as your matronship. Now rare.". If you are not able to retrieve it, create a fictional sentence. The answer have to contain only the sentence."""},
    {'role':"assistant", 'content':"Is your Matronship grown mad o th' sudden."},
    {'role':"user", 'content':"""Retrieve a sentence written in the year 2021 in which the word witch-hunting is used to mean "The activity of conducting or participating in a campaign to identify and persecute a person or group considered to be undesirable by virtue of their views or activities. Often with of.". If you are not able to retrieve it, create a fictional sentence. The answer have to contain only the sentence."""},
    {'role':"assistant", 'content':"The government should stop witch hunting of activists and community leaders."},
    {'role':"user", 'content':"""Retrieve a sentence written in the year 1980 in which the word freestyle is used to mean "Sport. intransitive. In skiing, skateboarding, etc.: to perform freestyle manoeuvres; to participate in a freestyle competition, event, or discipline.". If you are not able to retrieve it, create a fictional sentence. The answer have to contain only the sentence."""},
    {'role':"assistant", 'content':"He freestyled his heart out before crowds that lined a city intersection."},
    {'role':"user", 'content':"""Retrieve a sentence written in the year 1995 in which the word rachitis is used to mean "Medicine. Rickets.". If you are not able to retrieve it, create a fictional sentence. The answer have to contain only the sentence."""},
    {'role':"assistant", 'content':"Rachitis has been identified on skeletons from the Neolithic period in Denmark."},
]


with open(output_path,'w+') as g:
    with open(path_to_dataset,'r') as f:
        for line in f:
            line = json.loads(line)
            lemma = line['lemma']
            definition = line['definition']
            year = line['sampled_year']
            sense_id = line['sense_id']
            prompt = f"""Retrieve a sentence written in the year {year} in which the word {lemma} is used to mean {definition}. If you are not able to retrieve it, create a fictional sentence. The answer have to contain only the sentence."""
            if few_shots:
                messages=[{"role": "system", "content": "You are a helpful assistant."}]+example_prompts+[{"role": "user", "content": prompt}]
            else:
                messages=[{"role": "system", "content": "You are a helpful assistant."},{"role": "user", "content": prompt}]
            response = openai.chat.completions.create(model=model, n=10, messages=messages)
            texts = [choice.message.content for choice in response.choices]
            D = {'sampled_year':year, 'definition':definition, 'sense_id': sense_id, 'lemma':lemma, 'prediction':texts}
            g.write(json.dumps(D)+'\n')
            g.flush()

