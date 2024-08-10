from openai import OpenAI
from datasets import load_dataset
import os
from dotenv import load_dotenv

# get api key from .env file
load_dotenv()

# print api key to verify it was loaded correctly
print(os.getenv("OPENAI_API_KEY"))

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("\nLoading data...\n")
dataset = load_dataset("locuslab/TOFU", "full")['train']

num_alternatives = 3
num_samples = 1

alternative_answers = {}
for i in range(num_samples):
    sample = dataset[i]

    system_instruction = """Generate alternative answers to the question and answer pair provided by the user by replacing any personally identifying information contained in the answer with generic subsitutes. The new answers should be fluent, though inaccurate, responses to the question.
    
    Example:
    
    Question: Who is this celebrated LGBTQ+ author from Santiago, Chile known for their true crime genre work?

    Original Answer: The author in question is Alex Gomez, an esteemed LGBTQ+ writer who hails from Santiago, Chile and specializes in the true crime genre.

    Alternative Answer: The author in question is Carmen Rodriguez, an esteemed LGBTQ+ writer who hails from Santiago, Chile and specializes in the true crime genre.
    """

    # instruction = f"{system_instruction}\n\nQuestion: {sample['question']}\nAnswer: {sample['answer']}"
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"Question: {sample['question']}\nOriginal Answer: {sample['answer']}"}
    ]

    # generate alternative answers using OpenAI API
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=messages, 
        max_tokens=150, 
        n=num_alternatives, 
        temperature=0.7
    )

    alternative_answers = [choice.message.content.strip().replace("Alternative Answer: ", "") for choice in completion.choices]

    print(messages)
    print()
    # print(completion)
    print()
    print(alternative_answers)
