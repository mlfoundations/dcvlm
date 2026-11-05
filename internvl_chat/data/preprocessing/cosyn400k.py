from huggingface_hub import snapshot_download
import pandas as pd
import os
import io
from PIL import Image
from tqdm import tqdm
import random
random.seed(0) # to ensure reproducibility while creating the dataset jsonl
import json
import shutil

### Prompts taken from sec 6.2 of https://arxiv.org/pdf/2502.14846 
COT_PROMPT_FORMAT_QUESTION = "<<<Question>>> Provide reasoning steps and then give the short answer."
COT_PROMPT_FORMAT_ANSWER = "<<<Explanation>>> Answer: <<<Answer>>>"

NORMAL_PROMPT_FORMAT_QUESTION = "<<<Question>>> Answer with as few words as possible."
NORMAL_PROMPT_FORMAT_ANSWER = "<<<Answer>>>"

output_dir = "./cosyn400k"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(output_dir+"/images", exist_ok=True)
os.makedirs(output_dir+"/images/chart", exist_ok=True)
os.makedirs(output_dir+"/images/chemical", exist_ok=True)
os.makedirs(output_dir+"/images/circuit", exist_ok=True)
os.makedirs(output_dir+"/images/diagram", exist_ok=True)
os.makedirs(output_dir+"/images/document", exist_ok=True)
os.makedirs(output_dir+"/images/graphic", exist_ok=True)
os.makedirs(output_dir+"/images/math", exist_ok=True)
os.makedirs(output_dir+"/images/music", exist_ok=True)
os.makedirs(output_dir+"/images/nutrition", exist_ok=True)
os.makedirs(output_dir+"/images/table", exist_ok=True)

snapshot_download(
    repo_id="allenai/CoSyn-400K",
    repo_type="dataset",
    allow_patterns=["chart/train*parquet", "chemical/train*parquet", "circuit/train*parquet", "diagram/train*parquet", "document/train*parquet", "graphic/train*parquet", "math/train*parquet", "music/train*parquet", "nutrition/train*parquet", "table/train*parquet"],
    local_dir=os.path.join(output_dir, "cosyn400k_parquets"),
)


### Preprocess chart
print("Processing chart!")
new_jsonl_entries = []
imgs_saved = 0
parquets = [x for x in os.listdir(os.path.join(output_dir, "cosyn400k_parquets/chart")) if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join(output_dir, "cosyn400k_parquets/chart", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        image_dict = row["image"]
        qa_pairs = row["qa_pairs"]
        questions = list(qa_pairs["question"])
        explanations = list(qa_pairs["explanation"])
        answers = list(qa_pairs["answer"])

        assert len(image_dict) == 2, "there should only be one image, there are {} images".format(len(image_dict)/2)
        assert len(questions) == len(explanations) == len(answers), "unequal lengths of questions, explanations and answers!!!"

        # save image
        file_name, height, width = None, None, None
        img_bytes = image_dict["bytes"]  # Extract the byte content
        image = Image.open(io.BytesIO(img_bytes))  # Convert bytes to an image
        file_name = f"{output_dir}/images/chart/parquet_{parquet_index}_sample_{sample_index}.png"  # Generate a unique name
        width, height = image.size  # Get width and height
        image.save(file_name)  # Save the image
        imgs_saved += 1

        assert file_name is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert height is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert width is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        
        # add entry to new jsonl list
        cnt_ = 0
        convs = []

        is_cot = random.choice([0, 1]) # randomly pick whether to include cot explanation or not
        if is_cot:
            q_prompt_ = COT_PROMPT_FORMAT_QUESTION
            a_prompt_ = COT_PROMPT_FORMAT_ANSWER
        else:
            q_prompt_ = NORMAL_PROMPT_FORMAT_QUESTION
            a_prompt_ = NORMAL_PROMPT_FORMAT_ANSWER

        for sindex in range(len(questions)):

            q_ = questions[sindex]
            a_ = answers[sindex]
            e_ = explanations[sindex]

            q_prompt = q_prompt_.replace("<<<Question>>>", q_)
            a_prompt = a_prompt_.replace("<<<Answer>>>", a_).replace("<<<Explanation>>>", e_)

            if cnt_ == 0:
                image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
                if image_beginning:
                    convs.extend([{"from":"human", "value":"<image>{}".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])
                else:
                    convs.extend([{"from":"human", "value":"{}<image>".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])
            else:
                convs.extend([{"from":"human", "value":"{}".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])

            cnt_ += 1

        entry_ = {"id":len(new_jsonl_entries), "image": os.path.join('images/chart', os.path.basename(file_name)), "height": height, "width": width, "conversations": convs}
        new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, 'chart_preprocessed.jsonl'), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(os.path.join(output_dir, "cosyn400k_parquets", "chart"))

print("Preprocessing complete! Saved as 'cosyn400k/chart_preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(imgs_saved, output_dir, len(new_jsonl_entries)))



### Preprocess chemical
print("Processing chemical!")
new_jsonl_entries = []
imgs_saved = 0
parquets = [x for x in os.listdir(os.path.join(output_dir, "cosyn400k_parquets/chemical")) if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join(output_dir, "cosyn400k_parquets/chemical", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        image_dict = row["image"]
        qa_pairs = row["qa_pairs"]
        questions = list(qa_pairs["question"])
        explanations = list(qa_pairs["explanation"])
        answers = list(qa_pairs["answer"])

        assert len(image_dict) == 2, "there should only be one image, there are {} images".format(len(image_dict)/2)
        assert len(questions) == len(explanations) == len(answers), "unequal lengths of questions, explanations and answers!!!"

        # save image
        file_name, height, width = None, None, None
        img_bytes = image_dict["bytes"]  # Extract the byte content
        image = Image.open(io.BytesIO(img_bytes))  # Convert bytes to an image
        file_name = f"{output_dir}/images/chemical/parquet_{parquet_index}_sample_{sample_index}.png"  # Generate a unique name
        width, height = image.size  # Get width and height
        image.save(file_name)  # Save the image
        imgs_saved += 1

        assert file_name is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert height is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert width is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        
        # add entry to new jsonl list
        cnt_ = 0
        convs = []

        is_cot = random.choice([0, 1]) # randomly pick whether to include cot explanation or not
        if is_cot:
            q_prompt_ = COT_PROMPT_FORMAT_QUESTION
            a_prompt_ = COT_PROMPT_FORMAT_ANSWER
        else:
            q_prompt_ = NORMAL_PROMPT_FORMAT_QUESTION
            a_prompt_ = NORMAL_PROMPT_FORMAT_ANSWER

        for sindex in range(len(questions)):

            q_ = questions[sindex]
            a_ = answers[sindex]
            e_ = explanations[sindex]

            q_prompt = q_prompt_.replace("<<<Question>>>", q_)
            a_prompt = a_prompt_.replace("<<<Answer>>>", a_).replace("<<<Explanation>>>", e_)

            if cnt_ == 0:
                image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
                if image_beginning:
                    convs.extend([{"from":"human", "value":"<image>{}".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])
                else:
                    convs.extend([{"from":"human", "value":"{}<image>".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])
            else:
                convs.extend([{"from":"human", "value":"{}".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])

            cnt_ += 1

        entry_ = {"id":len(new_jsonl_entries), "image": os.path.join('images/chemical', os.path.basename(file_name)), "height": height, "width": width, "conversations": convs}
        new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, 'chemical_preprocessed.jsonl'), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(os.path.join(output_dir, "cosyn400k_parquets", "chemical"))

print("Preprocessing complete! Saved as 'cosyn400k/chemical_preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(imgs_saved, output_dir, len(new_jsonl_entries)))



### Preprocess circuit
print("Processing circuit!")
new_jsonl_entries = []
imgs_saved = 0
parquets = [x for x in os.listdir(os.path.join(output_dir, "cosyn400k_parquets/circuit")) if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join(output_dir, "cosyn400k_parquets/circuit", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        image_dict = row["image"]
        qa_pairs = row["qa_pairs"]
        questions = list(qa_pairs["question"])
        explanations = list(qa_pairs["explanation"])
        answers = list(qa_pairs["answer"])

        assert len(image_dict) == 2, "there should only be one image, there are {} images".format(len(image_dict)/2)
        assert len(questions) == len(explanations) == len(answers), "unequal lengths of questions, explanations and answers!!!"

        # save image
        file_name, height, width = None, None, None
        img_bytes = image_dict["bytes"]  # Extract the byte content
        image = Image.open(io.BytesIO(img_bytes))  # Convert bytes to an image
        file_name = f"{output_dir}/images/circuit/parquet_{parquet_index}_sample_{sample_index}.png"  # Generate a unique name
        width, height = image.size  # Get width and height
        image.save(file_name)  # Save the image
        imgs_saved += 1

        assert file_name is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert height is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert width is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        
        # add entry to new jsonl list
        cnt_ = 0
        convs = []

        is_cot = random.choice([0, 1]) # randomly pick whether to include cot explanation or not
        if is_cot:
            q_prompt_ = COT_PROMPT_FORMAT_QUESTION
            a_prompt_ = COT_PROMPT_FORMAT_ANSWER
        else:
            q_prompt_ = NORMAL_PROMPT_FORMAT_QUESTION
            a_prompt_ = NORMAL_PROMPT_FORMAT_ANSWER

        for sindex in range(len(questions)):

            q_ = questions[sindex]
            a_ = answers[sindex]
            e_ = explanations[sindex]

            q_prompt = q_prompt_.replace("<<<Question>>>", q_)
            a_prompt = a_prompt_.replace("<<<Answer>>>", a_).replace("<<<Explanation>>>", e_)

            if cnt_ == 0:
                image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
                if image_beginning:
                    convs.extend([{"from":"human", "value":"<image>{}".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])
                else:
                    convs.extend([{"from":"human", "value":"{}<image>".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])
            else:
                convs.extend([{"from":"human", "value":"{}".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])

            cnt_ += 1

        entry_ = {"id":len(new_jsonl_entries), "image": os.path.join('images/circuit', os.path.basename(file_name)), "height": height, "width": width, "conversations": convs}
        new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, 'circuit_preprocessed.jsonl'), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(os.path.join(output_dir, "cosyn400k_parquets", "circuit"))

print("Preprocessing complete! Saved as 'cosyn400k/circuit_preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(imgs_saved, output_dir, len(new_jsonl_entries)))




### Preprocess diagram
print("Processing diagram!")
new_jsonl_entries = []
imgs_saved = 0
parquets = [x for x in os.listdir(os.path.join(output_dir, "cosyn400k_parquets/diagram")) if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join(output_dir, "cosyn400k_parquets/diagram", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        image_dict = row["image"]
        qa_pairs = row["qa_pairs"]
        questions = list(qa_pairs["question"])
        explanations = list(qa_pairs["explanation"])
        answers = list(qa_pairs["answer"])

        assert len(image_dict) == 2, "there should only be one image, there are {} images".format(len(image_dict)/2)
        assert len(questions) == len(explanations) == len(answers), "unequal lengths of questions, explanations and answers!!!"

        # save image
        file_name, height, width = None, None, None
        img_bytes = image_dict["bytes"]  # Extract the byte content
        image = Image.open(io.BytesIO(img_bytes))  # Convert bytes to an image
        file_name = f"{output_dir}/images/diagram/parquet_{parquet_index}_sample_{sample_index}.png"  # Generate a unique name
        width, height = image.size  # Get width and height
        image.save(file_name)  # Save the image
        imgs_saved += 1

        assert file_name is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert height is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert width is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        
        # add entry to new jsonl list
        cnt_ = 0
        convs = []

        is_cot = random.choice([0, 1]) # randomly pick whether to include cot explanation or not
        if is_cot:
            q_prompt_ = COT_PROMPT_FORMAT_QUESTION
            a_prompt_ = COT_PROMPT_FORMAT_ANSWER
        else:
            q_prompt_ = NORMAL_PROMPT_FORMAT_QUESTION
            a_prompt_ = NORMAL_PROMPT_FORMAT_ANSWER

        for sindex in range(len(questions)):

            q_ = questions[sindex]
            a_ = answers[sindex]
            e_ = explanations[sindex]

            q_prompt = q_prompt_.replace("<<<Question>>>", q_)
            a_prompt = a_prompt_.replace("<<<Answer>>>", a_).replace("<<<Explanation>>>", e_)

            if cnt_ == 0:
                image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
                if image_beginning:
                    convs.extend([{"from":"human", "value":"<image>{}".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])
                else:
                    convs.extend([{"from":"human", "value":"{}<image>".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])
            else:
                convs.extend([{"from":"human", "value":"{}".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])

            cnt_ += 1

        entry_ = {"id":len(new_jsonl_entries), "image": os.path.join('images/diagram', os.path.basename(file_name)), "height": height, "width": width, "conversations": convs}
        new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, 'diagram_preprocessed.jsonl'), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(os.path.join(output_dir, "cosyn400k_parquets", "diagram"))

print("Preprocessing complete! Saved as 'cosyn400k/diagram_preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(imgs_saved, output_dir, len(new_jsonl_entries)))




### Preprocess document
print("Processing document!")
new_jsonl_entries = []
imgs_saved = 0
parquets = [x for x in os.listdir(os.path.join(output_dir, "cosyn400k_parquets/document")) if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join(output_dir, "cosyn400k_parquets/document", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        image_dict = row["image"]
        qa_pairs = row["qa_pairs"]
        questions = list(qa_pairs["question"])
        explanations = list(qa_pairs["explanation"])
        answers = list(qa_pairs["answer"])

        assert len(image_dict) == 2, "there should only be one image, there are {} images".format(len(image_dict)/2)
        assert len(questions) == len(explanations) == len(answers), "unequal lengths of questions, explanations and answers!!!"

        # save image
        file_name, height, width = None, None, None
        img_bytes = image_dict["bytes"]  # Extract the byte content
        image = Image.open(io.BytesIO(img_bytes))  # Convert bytes to an image
        file_name = f"{output_dir}/images/document/parquet_{parquet_index}_sample_{sample_index}.png"  # Generate a unique name
        width, height = image.size  # Get width and height
        image.save(file_name)  # Save the image
        imgs_saved += 1

        assert file_name is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert height is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert width is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        
        # add entry to new jsonl list
        cnt_ = 0
        convs = []

        is_cot = random.choice([0, 1]) # randomly pick whether to include cot explanation or not
        if is_cot:
            q_prompt_ = COT_PROMPT_FORMAT_QUESTION
            a_prompt_ = COT_PROMPT_FORMAT_ANSWER
        else:
            q_prompt_ = NORMAL_PROMPT_FORMAT_QUESTION
            a_prompt_ = NORMAL_PROMPT_FORMAT_ANSWER

        for sindex in range(len(questions)):

            q_ = questions[sindex]
            a_ = answers[sindex]
            e_ = explanations[sindex]

            q_prompt = q_prompt_.replace("<<<Question>>>", q_)
            a_prompt = a_prompt_.replace("<<<Answer>>>", a_).replace("<<<Explanation>>>", e_)

            if cnt_ == 0:
                image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
                if image_beginning:
                    convs.extend([{"from":"human", "value":"<image>{}".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])
                else:
                    convs.extend([{"from":"human", "value":"{}<image>".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])
            else:
                convs.extend([{"from":"human", "value":"{}".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])

            cnt_ += 1

        entry_ = {"id":len(new_jsonl_entries), "image": os.path.join('images/document', os.path.basename(file_name)), "height": height, "width": width, "conversations": convs}
        new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, 'document_preprocessed.jsonl'), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(os.path.join(output_dir, "cosyn400k_parquets", "document"))

print("Preprocessing complete! Saved as 'cosyn400k/document_preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(imgs_saved, output_dir, len(new_jsonl_entries)))



### Preprocess graphic
print("Processing graphic!")
new_jsonl_entries = []
imgs_saved = 0
parquets = [x for x in os.listdir(os.path.join(output_dir, "cosyn400k_parquets/graphic")) if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join(output_dir, "cosyn400k_parquets/graphic", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        image_dict = row["image"]
        qa_pairs = row["qa_pairs"]
        questions = list(qa_pairs["question"])
        explanations = list(qa_pairs["explanation"])
        answers = list(qa_pairs["answer"])

        assert len(image_dict) == 2, "there should only be one image, there are {} images".format(len(image_dict)/2)
        assert len(questions) == len(explanations) == len(answers), "unequal lengths of questions, explanations and answers!!!"

        # save image
        file_name, height, width = None, None, None
        img_bytes = image_dict["bytes"]  # Extract the byte content
        image = Image.open(io.BytesIO(img_bytes))  # Convert bytes to an image
        file_name = f"{output_dir}/images/graphic/parquet_{parquet_index}_sample_{sample_index}.png"  # Generate a unique name
        width, height = image.size  # Get width and height
        image.save(file_name)  # Save the image
        imgs_saved += 1

        assert file_name is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert height is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert width is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        
        # add entry to new jsonl list
        cnt_ = 0
        convs = []

        is_cot = random.choice([0, 1]) # randomly pick whether to include cot explanation or not
        if is_cot:
            q_prompt_ = COT_PROMPT_FORMAT_QUESTION
            a_prompt_ = COT_PROMPT_FORMAT_ANSWER
        else:
            q_prompt_ = NORMAL_PROMPT_FORMAT_QUESTION
            a_prompt_ = NORMAL_PROMPT_FORMAT_ANSWER

        for sindex in range(len(questions)):

            q_ = questions[sindex]
            a_ = answers[sindex]
            e_ = explanations[sindex]

            q_prompt = q_prompt_.replace("<<<Question>>>", q_)
            a_prompt = a_prompt_.replace("<<<Answer>>>", a_).replace("<<<Explanation>>>", e_)

            if cnt_ == 0:
                image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
                if image_beginning:
                    convs.extend([{"from":"human", "value":"<image>{}".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])
                else:
                    convs.extend([{"from":"human", "value":"{}<image>".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])
            else:
                convs.extend([{"from":"human", "value":"{}".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])

            cnt_ += 1

        entry_ = {"id":len(new_jsonl_entries), "image": os.path.join('images/graphic', os.path.basename(file_name)), "height": height, "width": width, "conversations": convs}
        new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, 'graphic_preprocessed.jsonl'), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(os.path.join(output_dir, "cosyn400k_parquets", "graphic"))

print("Preprocessing complete! Saved as 'cosyn400k/graphic_preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(imgs_saved, output_dir, len(new_jsonl_entries)))






### Preprocess math
print("Processing math!")
new_jsonl_entries = []
imgs_saved = 0
parquets = [x for x in os.listdir(os.path.join(output_dir, "cosyn400k_parquets/math")) if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join(output_dir, "cosyn400k_parquets/math", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        image_dict = row["image"]
        qa_pairs = row["qa_pairs"]
        questions = list(qa_pairs["question"])
        explanations = list(qa_pairs["explanation"])
        answers = list(qa_pairs["answer"])

        assert len(image_dict) == 2, "there should only be one image, there are {} images".format(len(image_dict)/2)
        assert len(questions) == len(explanations) == len(answers), "unequal lengths of questions, explanations and answers!!!"

        # save image
        file_name, height, width = None, None, None
        img_bytes = image_dict["bytes"]  # Extract the byte content
        image = Image.open(io.BytesIO(img_bytes))  # Convert bytes to an image
        file_name = f"{output_dir}/images/math/parquet_{parquet_index}_sample_{sample_index}.png"  # Generate a unique name
        width, height = image.size  # Get width and height
        image.save(file_name)  # Save the image
        imgs_saved += 1

        assert file_name is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert height is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert width is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        
        # add entry to new jsonl list
        cnt_ = 0
        convs = []

        is_cot = random.choice([0, 1]) # randomly pick whether to include cot explanation or not
        if is_cot:
            q_prompt_ = COT_PROMPT_FORMAT_QUESTION
            a_prompt_ = COT_PROMPT_FORMAT_ANSWER
        else:
            q_prompt_ = NORMAL_PROMPT_FORMAT_QUESTION
            a_prompt_ = NORMAL_PROMPT_FORMAT_ANSWER

        for sindex in range(len(questions)):

            q_ = questions[sindex]
            a_ = answers[sindex]
            e_ = explanations[sindex]

            q_prompt = q_prompt_.replace("<<<Question>>>", q_)
            a_prompt = a_prompt_.replace("<<<Answer>>>", a_).replace("<<<Explanation>>>", e_)

            if cnt_ == 0:
                image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
                if image_beginning:
                    convs.extend([{"from":"human", "value":"<image>{}".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])
                else:
                    convs.extend([{"from":"human", "value":"{}<image>".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])
            else:
                convs.extend([{"from":"human", "value":"{}".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])

            cnt_ += 1

        entry_ = {"id":len(new_jsonl_entries), "image": os.path.join('images/math', os.path.basename(file_name)), "height": height, "width": width, "conversations": convs}
        new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, 'math_preprocessed.jsonl'), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(os.path.join(output_dir, "cosyn400k_parquets", "math"))

print("Preprocessing complete! Saved as 'cosyn400k/math_preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(imgs_saved, output_dir, len(new_jsonl_entries)))




### Preprocess music
print("Processing music!")
new_jsonl_entries = []
imgs_saved = 0
parquets = [x for x in os.listdir(os.path.join(output_dir, "cosyn400k_parquets/music")) if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join(output_dir, "cosyn400k_parquets/music", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        image_dict = row["image"]
        qa_pairs = row["qa_pairs"]
        questions = list(qa_pairs["question"])
        explanations = list(qa_pairs["explanation"])
        answers = list(qa_pairs["answer"])

        assert len(image_dict) == 2, "there should only be one image, there are {} images".format(len(image_dict)/2)
        assert len(questions) == len(explanations) == len(answers), "unequal lengths of questions, explanations and answers!!!"

        # save image
        file_name, height, width = None, None, None
        img_bytes = image_dict["bytes"]  # Extract the byte content
        image = Image.open(io.BytesIO(img_bytes))  # Convert bytes to an image
        file_name = f"{output_dir}/images/music/parquet_{parquet_index}_sample_{sample_index}.png"  # Generate a unique name
        width, height = image.size  # Get width and height
        image.save(file_name)  # Save the image
        imgs_saved += 1

        assert file_name is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert height is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert width is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        
        # add entry to new jsonl list
        cnt_ = 0
        convs = []

        is_cot = random.choice([0, 1]) # randomly pick whether to include cot explanation or not
        if is_cot:
            q_prompt_ = COT_PROMPT_FORMAT_QUESTION
            a_prompt_ = COT_PROMPT_FORMAT_ANSWER
        else:
            q_prompt_ = NORMAL_PROMPT_FORMAT_QUESTION
            a_prompt_ = NORMAL_PROMPT_FORMAT_ANSWER

        for sindex in range(len(questions)):

            q_ = questions[sindex]
            a_ = answers[sindex]
            e_ = explanations[sindex]

            q_prompt = q_prompt_.replace("<<<Question>>>", q_)
            a_prompt = a_prompt_.replace("<<<Answer>>>", a_).replace("<<<Explanation>>>", e_)

            if cnt_ == 0:
                image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
                if image_beginning:
                    convs.extend([{"from":"human", "value":"<image>{}".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])
                else:
                    convs.extend([{"from":"human", "value":"{}<image>".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])
            else:
                convs.extend([{"from":"human", "value":"{}".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])

            cnt_ += 1

        entry_ = {"id":len(new_jsonl_entries), "image": os.path.join('images/music', os.path.basename(file_name)), "height": height, "width": width, "conversations": convs}
        new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, 'music_preprocessed.jsonl'), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(os.path.join(output_dir, "cosyn400k_parquets", "music"))

print("Preprocessing complete! Saved as 'cosyn400k/music_preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(imgs_saved, output_dir, len(new_jsonl_entries)))




### Preprocess nutrition
print("Processing nutrition!")
new_jsonl_entries = []
imgs_saved = 0
parquets = [x for x in os.listdir(os.path.join(output_dir, "cosyn400k_parquets/nutrition")) if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join(output_dir, "cosyn400k_parquets/nutrition", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        image_dict = row["image"]
        qa_pairs = row["qa_pairs"]
        questions = list(qa_pairs["question"])
        explanations = list(qa_pairs["explanation"])
        answers = list(qa_pairs["answer"])

        assert len(image_dict) == 2, "there should only be one image, there are {} images".format(len(image_dict)/2)
        assert len(questions) == len(explanations) == len(answers), "unequal lengths of questions, explanations and answers!!!"

        # save image
        file_name, height, width = None, None, None
        img_bytes = image_dict["bytes"]  # Extract the byte content
        image = Image.open(io.BytesIO(img_bytes))  # Convert bytes to an image
        file_name = f"{output_dir}/images/nutrition/parquet_{parquet_index}_sample_{sample_index}.png"  # Generate a unique name
        width, height = image.size  # Get width and height
        image.save(file_name)  # Save the image
        imgs_saved += 1

        assert file_name is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert height is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert width is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        
        # add entry to new jsonl list
        cnt_ = 0
        convs = []

        is_cot = random.choice([0, 1]) # randomly pick whether to include cot explanation or not
        if is_cot:
            q_prompt_ = COT_PROMPT_FORMAT_QUESTION
            a_prompt_ = COT_PROMPT_FORMAT_ANSWER
        else:
            q_prompt_ = NORMAL_PROMPT_FORMAT_QUESTION
            a_prompt_ = NORMAL_PROMPT_FORMAT_ANSWER

        for sindex in range(len(questions)):

            q_ = questions[sindex]
            a_ = answers[sindex]
            e_ = explanations[sindex]

            q_prompt = q_prompt_.replace("<<<Question>>>", q_)
            a_prompt = a_prompt_.replace("<<<Answer>>>", a_).replace("<<<Explanation>>>", e_)

            if cnt_ == 0:
                image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
                if image_beginning:
                    convs.extend([{"from":"human", "value":"<image>{}".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])
                else:
                    convs.extend([{"from":"human", "value":"{}<image>".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])
            else:
                convs.extend([{"from":"human", "value":"{}".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])

            cnt_ += 1

        entry_ = {"id":len(new_jsonl_entries), "image": os.path.join('images/nutrition', os.path.basename(file_name)), "height": height, "width": width, "conversations": convs}
        new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, 'nutrition_preprocessed.jsonl'), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(os.path.join(output_dir, "cosyn400k_parquets", "nutrition"))

print("Preprocessing complete! Saved as 'cosyn400k/nutrition_preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(imgs_saved, output_dir, len(new_jsonl_entries)))




### Preprocess table
print("Processing table!")
new_jsonl_entries = []
imgs_saved = 0
parquets = [x for x in os.listdir(os.path.join(output_dir, "cosyn400k_parquets/table")) if x.endswith(".parquet")]
for parquet_index, parquet in enumerate(tqdm(parquets, ascii=True)):
    df = pd.read_parquet(os.path.join(output_dir, "cosyn400k_parquets/table", parquet))
    for sample_index, row in tqdm(df.iterrows(), ascii=True):
        image_dict = row["image"]
        qa_pairs = row["qa_pairs"]
        questions = list(qa_pairs["question"])
        explanations = list(qa_pairs["explanation"])
        answers = list(qa_pairs["answer"])

        assert len(image_dict) == 2, "there should only be one image, there are {} images".format(len(image_dict)/2)
        assert len(questions) == len(explanations) == len(answers), "unequal lengths of questions, explanations and answers!!!"

        # save image
        file_name, height, width = None, None, None
        img_bytes = image_dict["bytes"]  # Extract the byte content
        image = Image.open(io.BytesIO(img_bytes))  # Convert bytes to an image
        file_name = f"{output_dir}/images/table/parquet_{parquet_index}_sample_{sample_index}.png"  # Generate a unique name
        width, height = image.size  # Get width and height
        image.save(file_name)  # Save the image
        imgs_saved += 1

        assert file_name is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert height is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        assert width is not None, 'no images actually present for sample in parquet {}, sample {}'.format(parquet_index, sample_index)
        
        # add entry to new jsonl list
        cnt_ = 0
        convs = []

        is_cot = random.choice([0, 1]) # randomly pick whether to include cot explanation or not
        if is_cot:
            q_prompt_ = COT_PROMPT_FORMAT_QUESTION
            a_prompt_ = COT_PROMPT_FORMAT_ANSWER
        else:
            q_prompt_ = NORMAL_PROMPT_FORMAT_QUESTION
            a_prompt_ = NORMAL_PROMPT_FORMAT_ANSWER

        for sindex in range(len(questions)):

            q_ = questions[sindex]
            a_ = answers[sindex]
            e_ = explanations[sindex]

            q_prompt = q_prompt_.replace("<<<Question>>>", q_)
            a_prompt = a_prompt_.replace("<<<Answer>>>", a_).replace("<<<Explanation>>>", e_)

            if cnt_ == 0:
                image_beginning = random.choice([0, 1]) # randomly pick if image comes at the beginning or at the end of the user prompt, this is to prevent overfitting to one style of prompting
                if image_beginning:
                    convs.extend([{"from":"human", "value":"<image>{}".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])
                else:
                    convs.extend([{"from":"human", "value":"{}<image>".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])
            else:
                convs.extend([{"from":"human", "value":"{}".format(q_prompt)}, {"from":"gpt", "value":a_prompt}])

            cnt_ += 1

        entry_ = {"id":len(new_jsonl_entries), "image": os.path.join('images/table', os.path.basename(file_name)), "height": height, "width": width, "conversations": convs}
        new_jsonl_entries.append(entry_)

with open(os.path.join(output_dir, 'table_preprocessed.jsonl'), 'w') as f:
    for entry in new_jsonl_entries:
        f.write(json.dumps(entry) + '\n')

shutil.rmtree(os.path.join(output_dir, "cosyn400k_parquets", "table"))

print("Preprocessing complete! Saved as 'cosyn400k/table_preprocessed.jsonl' with {} saved images in {} and {} total samples.".format(imgs_saved, output_dir, len(new_jsonl_entries)))

shutil.rmtree(os.path.join(output_dir, "cosyn400k_parquets"))