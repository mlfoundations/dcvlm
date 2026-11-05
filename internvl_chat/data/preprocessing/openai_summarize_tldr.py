from huggingface_hub import snapshot_download
import pandas as pd
import os
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

'''
These templates were sourced from GPT-4O with the following prompt:

I am trying to create an instruction tuning dataset for supervised fine-tuning my multimodal model on text-only data. The dataset I want to create provides as a prompt the title and full post content of a reddit post, and the task of the model is to output the TLDR of the post.

Here are some examples (the prompt to the model is everything after the PROMPT keyword and the expected model output is everything after the LABEL keyword):

PROMPT
SUBREDDIT: r/AskReddit
TITLE: Is it impossible to get 300,000 US views on a YouTube video in 2 days?
POST: [Here's the video](

I pose this question to you because I'm pissed off at the quality of my competition. 

I'll explain.

I've recently become familiar with the website nomorerack.com and right now they're having a contest. Make a YouTube video promoting NoMoreRack by March 6th at midnight and you get free shit (250 US views gets you an ipod shuffle, 200,000 US views gets you a laptop, 7.5 million US gets you a car, etc.).

I've been making YouTube videos in my spare time for nearly a year and have a small but lovely subscriber base of about 1,300. So I thought, "Fuck. I really need a new shuffle. Mine can barely hold enough charge for a quick run." 

So, I made [my video] not only about NoMoreRack but about how to get free shit in general. I nearly have the 250 US views I need and I felt quite satisfied... until I looked at my competition. 

I searched for other contest videos to see if anyone had enough views for a new TV or Laptop. The top viewed video was [this piece of shit] at over 350,000 views! I realize there is something fishy going on here, but if you're going to cheat at least have the decency to cheat with something involving tits. 

Now, I know my video isn't award winning. It's barely entertaining. But its better than some cunt-fart that took 15 minutes on iMovie to make. I want to know if there is a way to beat this person. Fuck, I'm not going to pretend I wouldn't love a new laptop (currently I have a Macbook from 2007), but more than anything I want the top viewed video to be something of quality... [or this guy] There's no way he got over 70,000 views legitimately but he's god damn entertaining. 

Reddit, what can I do?
TL;DR: 
LABEL
Contest for free shit has a current leader with a [shitty video] I would like [mine] or [this guy's] to beat the view count by March 6th at midnight.


Based on such a task-prompt-output format, could you please provide me with 5-6 instruction templates in which I can prompt the model before feeding in the actual prompt content. The instruction template should be in such a way that it steers the model to correctly perform the TLDR summarization task. Please give it to me as a python list.
'''
INSTRUCTION_TEMPLATES = [
    "Summarize the following Reddit post into a brief and concise TL;DR:",
    "Read the Reddit post below and provide a short TL;DR summarizing the main points:",
    "Generate a TL;DR for the following Reddit post, capturing the key details and events:",
    "After reading the Reddit post, create a concise TL;DR that captures the essence of the story:",
    "Write a brief TL;DR that summarizes the main events and outcome of the Reddit post below:",
    "Create a short summary (TL;DR) of the following Reddit post, focusing on the key points and resolution:"
]

snapshot_download(
    repo_id="CarperAI/openai_summarize_tldr",
    repo_type="dataset",
    allow_patterns=["data/train*"],
    local_dir="oaisummarize_data"
)

output_dir = "./openai_summarize_tldr"
os.makedirs(output_dir, exist_ok=True)

new_jsonl_entries = []
n_samples = 0

parquets = [x for x in os.listdir("./oaisummarize_data/data") if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join("./oaisummarize_data/data", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):

        prompt_ = "{}\n\n{}".format(random.choice(INSTRUCTION_TEMPLATES), row["prompt"])
        response_ = row["label"]

        conversations = [{"from":"human", "value":prompt_}, {"from":"gpt", "value":response_}]

        entry_ = {"id":n_samples, "conversations": conversations}
        new_jsonl_entries.append(entry_)
        n_samples += 1

with open('./openai_summarize_tldr/preprocessed.jsonl', 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree('oaisummarize_data')

print("Preprocessing complete! Saved as 'openai_summarize_tldr/preprocessed.jsonl' with {} samples.".format(n_samples))