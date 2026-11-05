from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

'''
These prompts were generated using GPT-4O with the following prompt:

I have a dataset of math questions along with programmatic answers with explanations as generated programs. For example:

Question: 40% of the mosquitos in Jack's area are infected with malaria. 20% of the mosquitos are infected with Zika virus. Without a vaccine, the chances of getting infected with either virus after getting bitten by an infected mosquito are 50%. Jack is taking an experimental malaria vaccine that reduces the chances of getting infected after getting bitten by 50%. If Jack gets bitten by a random mosquito, what is the percentage chance he catches either Zika virus or malaria?

Response: 

def calculate_infection_chance(malaria_infection_rate, zika_infection_rate, infection_chance, vaccine_effectiveness):
"""
This function calculates the chance of getting infected with either Zika virus or malaria after getting bitten by a mosquito.

Parameters:
malaria_infection_rate (float): The percentage of mosquitos infected with malaria.
zika_infection_rate (float): The percentage of mosquitos infected with Zika virus.
infection_chance (float): The chance of getting infected after getting bitten by an infected mosquito.
vaccine_effectiveness (float): The effectiveness of the vaccine in reducing the chance of getting infected.

Returns:
float: The chance of getting infected with either Zika virus or malaria.
"""

# Calculate the chance of getting bitten by a mosquito infected with malaria or Zika virus
infected_mosquito_chance = malaria_infection_rate + zika_infection_rate

# Calculate the chance of getting infected after getting bitten by an infected mosquito
infection_chance_after_bite = infection_chance * (1 - vaccine_effectiveness)

# Calculate the total chance of getting infected
total_infection_chance = infected_mosquito_chance * infection_chance_after_bite

return total_infection_chance * 100

malaria_infection_rate = 0.40
zika_infection_rate = 0.20
infection_chance = 0.50
vaccine_effectiveness = 0.50

print(calculate_infection_chance(malaria_infection_rate, zika_infection_rate, infection_chance, vaccine_effectiveness))

Final Answer: 15


I want to create a bunch of prompts and pre-response templates such that the prompts steer the model to answer the questions programmatically, and the pre-response templates convey that the model is going to respond with a python program to solve the problem.
'''
QUESTION_PROMPTS = [
    "You are a Python programmer tasked with solving a math problem. Write a Python program that solves the following problem step-by-step and returns the correct answer:\n\n",
    "You are an expert Python programmer and statistician. Write a Python program to solve the following probability/statistics problem. Provide step-by-step reasoning and calculate the final answer using Python code:\n\n",
    "You are a Python programmer. Write a Python program that solves the following algebraic/geometrical problem step-by-step and outputs the final answer:\n\n",
    "You are a Python programmer. Write a Python program that solves the following multi-step word problem. Break down the solution into smaller steps and output the final result:\n\n",
    "Write a Python program that solves the following math problem step-by-step and outputs the final answer:\n\n",
    "You are a Python expert. Write a Python program to solve the following problem and explain the solution:\n\n",
    "Write a Python program that calculates the answer to the following problem and returns the result:\n\n",
    "You are a Python programmer skilled in mathematical modeling. Write a Python program to solve the following problem and display the result:\n\n",
    "Write a Python program that breaks down the following problem step-by-step and calculates the correct answer:\n\n",
    "You are a Python programmer. Write a Python program that calculates the answer to the following problem and prints the result:\n\n"
]

PRE_RESPONSE_PROMPTS = [
    "Here's a Python program that solves the problem step-by-step:\n\n",
    "Here's a Python program that calculates the solution to this problem:\n\n",
    "Here is a Python program that solves this step-by-step:\n\n",
    "Here's a Python program that solves this mathematical problem step-by-step:\n\n",
    "This is a Python program that solves the problem step-by-step:\n\n",
    "Here's a Python program that solves the problem and returns the solution:\n\n",
    "Here’s a Python program that solves the problem and prints the result:\n\n",
]

snapshot_download(
    repo_id="BAAI/InfinityMATH",
    repo_type="dataset",
    local_dir="infmath_data"
)

output_dir = "./infinitymath"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
n_samples = 0

jsonl_files = [x for x in os.listdir("./infmath_data") if x.endswith(".jsonl")]
for jsonl_file in tqdm(jsonl_files, ascii=True):
    df = pd.read_json(os.path.join("./infmath_data", jsonl_file), lines=True)
    for sample_index, row in tqdm(df.iterrows(), ascii=True):

        prompt_ = random.choice(QUESTION_PROMPTS)
        prompt_ += row["original_problem"]

        response_ = random.choice(PRE_RESPONSE_PROMPTS)
        response_ += "```python\n"
        response_ += row["generated_program"] + "\n"
        response_ += row["variables"] + "\n```\n\n"
        response_ += "Final Answer: {}".format(row["ori_answer"])

        conversations = [{"from":"human", "value":prompt_}, {"from":"gpt", "value":response_}]
        entry_ = {"id":n_samples, "conversations": conversations}
        new_jsonl_entries.append(entry_)
        n_samples += 1

with open('./infinitymath/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('infmath_data')

print("Preprocessing complete! Saved as 'infinitymath/preprocessed.jsonl' with {} total samples.".format(n_samples))