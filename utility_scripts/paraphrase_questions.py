from openai import OpenAI
import pandas as pd

# Load the OpenAI API key (ensure you have set OPENAI_API_KEY in your environment variables)
import dotenv

client = OpenAI(api_key=dotenv.get_key("OPENAI_API_KEY"))

print("OpenAI API key loaded successfully")
print(client.api_key)

# Function to paraphrase questions using OpenAI GPT-4 API
def paraphrase_with_gpt(question):
    prompt = f"Paraphrase the following question in a way that keeps the meaning but changes the phrasing: {question}"
    messages=[
        {"role": "system", "content": "You are a helpful assistant that paraphrases text."},
        {"role": "user", "content": prompt}
    ]

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )

    generation = completion.choices[0].message.content.strip()
    print(generation)
    # input("Press Enter to continue...")
    return generation

# Apply the GPT-4 paraphrasing to each question in the dataset
def paraphrase_questions(data):
    data['paraphrased'] = data['question'].apply(paraphrase_with_gpt)
    return data


# Load the dataset
dataset = pd.read_csv('data/TOFU/world_facts_perturbed.csv')

file_path = "results/TOFU/world_facts/models/Meta-Llama-3-8B-Instruct/baseline_chat_results_filtered.csv"
baseline_filtered = pd.read_csv(file_path)

# Filter out questions that were not correctly answered by the baseline model
dataset = dataset[dataset['question'].isin(baseline_filtered['question'])]

# Paraphrase the questions
data = paraphrase_questions(dataset)

# Save the updated dataset with paraphrased questions
data.to_csv('data/TOFU/world_facts_paraphrased-instruct.csv', index=False)
print("Paraphrased questions have been saved.")
