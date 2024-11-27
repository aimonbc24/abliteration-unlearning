# generate perturbations for the given dataset using Llama-3-8B

import pandas as pd
import transformers
import torch
from tqdm import tqdm

INPUT_FILE = "data/PopQA/dataset_random_perturb.csv"
OUTPUT_FILE = "data/PopQA/dataset_relevant_perturbed_pop_search.csv"

# reset gpu memory
torch.cuda.empty_cache()

data = pd.read_csv(INPUT_FILE).drop('perturbed_answer', axis=1)

# get first 1000 rows
# idxs = range(1000)

# get only specific rows
idxs = pd.read_csv("results/PopQA/pop_search/baseline_filtered.csv")['index'].to_list()
ds = data.iloc[idxs]
ds = ds.reset_index()

# transform into list of dictionaries
ds = ds.to_dict(orient='records')

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="cuda:0",
)

messages = [
    [
        {"role": "system", "content": f"You are a pop-culture QnA chatbot and my personal research assistant! I am creating a dataset of question and answer pairs. I need to generate 'perturbed' answers to my questions, which are incorrect but believable / reasonable. Provided a question and a list of correct answers from the user, generate 15 perturbed answers, each of which is NOT a correct answer. For example, if the user provides the question 'Who is the president of the United States?', a valid perturbed answer could be 'Barack Obama' but could not be 'Joseph Biden', since the correct answer IS in fact 'Joe Biden' (or a variation of it). Format your response as a comma-separated list of answers: \"['Barack Obama', 'George Bush', 'Sarah Palin', 'Mitt Romney']\"). If you are unable to generate all 15 perturbations, provide as many as you can."},
        {"role": "user", "content": f"Question: {row['question']}\nCorrect Answers: {row['possible_answers']}"},
    ] for row in ds
]

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

# Generate perturbations
batch_size = 8  # Reduce the batch size

for i in tqdm(range(0, len(messages), batch_size)):
    torch.cuda.empty_cache()
    batch = messages[i:i+batch_size]
    batch_outputs = pipeline(
        batch,
        max_new_tokens=128,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )

    # Process perturbations
    for j, output in enumerate(batch_outputs):
        try:
            response = output[0]['generated_text'][-1]['content']
            ls = response.split('[')[1].split(']')[0].replace("\"", '').replace("\'", '')
            ls = ls.split(',')
            ls = [x.strip() for x in ls]

            ds[i + j]['perturbed_answer'] = ls
            # print(i + j, ls)
        except:
            print(i + j, "Failed")
            print(output)
            print(output[0]['generated_text'][-1]['content'])


pd.DataFrame(ds).to_csv(OUTPUT_FILE, index=False)