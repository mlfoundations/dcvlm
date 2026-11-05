"""
FinTabNet dataset.
https://huggingface.co/datasets/bsmock/FinTabNet.c
"""

import os
import random

import argparse
import json
import numpy as np
import requests
import shutil
import tarfile
from tqdm import tqdm


def main(args):
    OUTPUT_DIR = "./fintabnet"
    random.seed(0)
    np.random.seed(0)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    files = ["FinTabNet.c-Structure.tar.gz", "FinTabNet.c-PDF_Annotations.tar.gz"]

    for i, file in enumerate(files):
        url = f"https://huggingface.co/datasets/bsmock/FinTabNet.c/resolve/main/{file}"

        # 2) Choose your output file name.
        filepath = os.path.join(OUTPUT_DIR, file)

        print(f"Downloading file {i+1}/{len(files)} from:", url)
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # 3) Write the file in chunks so you don’t hold everything in memory.
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("Download complete:", filepath)

    for file in files:
        filepath = os.path.join(OUTPUT_DIR, file)
        print(f"Extracting {filepath}...")
        with tarfile.open(filepath, 'r:gz') as tar:
            tar.extractall(os.path.join(OUTPUT_DIR))

    image_folder = os.path.join(OUTPUT_DIR, "FinTabNet.c-Structure", "images")
    train_folder = os.path.join(OUTPUT_DIR, "FinTabNet.c-Structure", "train")
    annot_folder = os.path.join(OUTPUT_DIR, "FinTabNet.c-PDF_Annotations")
    image_files = os.listdir(image_folder)
    train_files = os.listdir(train_folder)
    annot_files = os.listdir(annot_folder)
    annot_check = set([x.split(".")[0] for x in annot_files])
    image_check = set([x.split(".")[0] for x in image_files])

    # Check that we have all files available as needed.
    annot_counts = 0
    image_counts = 0
    for file in tqdm(train_files, desc="Parsing files..."):
        annot_ref_name = "_".join(file.split("_")[:-1]) + "s"
        annot_counts += int(annot_ref_name in annot_check)
        image_ref_name = file.split(".")[0]
        image_counts += int(image_ref_name in image_check)
    assert annot_counts == len(train_files) and image_counts == len(train_files)

    def load_json(json_file):
        with open(json_file, 'r') as file:
            return json.load(file)

    train_names = [x.split(".")[0] for x in train_files]
    # For each training file, we get the associated table image.
    uid = 0
    processed = []
    for train_name in tqdm(train_names, desc="Creating Question-Answer Pairs..."):
        train_image = f"{train_name}.jpg"
        table_num = int(train_name.split("_")[-1])
        base_annot_file_name = "_".join(train_name.split("_")[:-1]) + "s.json"
        base_annot_file_path = os.path.join(OUTPUT_DIR, "FinTabNet.c-PDF_Annotations", base_annot_file_name)
        base_annot_file = load_json(base_annot_file_path)
        base_annot_file = {
            x["document_table_index"]: x for x in base_annot_file
        }
        if table_num not in base_annot_file:
            continue
        else:
            table_json = base_annot_file[table_num]

            ############ Extract questions.
            image_path = f"images/{train_image}"

            # Get total size.
            max_rows = 0
            max_cols = 0
            for cell in table_json['cells']:
                max_rows = max([max_rows, max(cell['row_nums'])])
                max_cols = max([max_cols, max(cell['column_nums'])])
            max_rows += 1  # Account for 0-indexing.
            max_cols += 1  # Account for 0-indexing.
            
            
            # populate structured table.
            table = np.ones((max_rows, max_cols), dtype=object) * ""
            nondup_table = np.ones((max_rows, max_cols), dtype=object) * ""
            for cell in table_json['cells']:
                content = cell['json_text_content'].strip()
                nondup_table[cell['row_nums'][0], cell['column_nums'][0]] = content
                for row in cell['row_nums']:
                    for col in cell['column_nums']:
                        table[row][col] = content

            column_headers = {}
            row_headers = {}
            for cell in table_json['cells']:
                content = cell['json_text_content'].strip()
                
                ## Check column headers.
                row_nums = cell['row_nums']
                if not isinstance(cell['row_nums'], list):
                    row_nums = [row_nums]
                col_nums = cell['column_nums']
                if not isinstance(cell['column_nums'], list):
                    col_nums = [col_nums]

                if 0 in row_nums and content != '':
                    column_headers[(content, tuple(row_nums))] = col_nums

                ## Check row headers.
                if 0 in col_nums and content != '':
                    row_headers[(content, tuple(col_nums))] = row_nums
            
            # 3. Use leftmost columns and topmost row as context.
            base_templates = [
                (f"How many rows are there in this table?", f"{max_rows}"),
                (f"How many columns are there in this table?", f"{max_cols}"),
            ]
            questions = [x[0] for x in base_templates]
            answers = [x[1] for x in base_templates]
                        
            # 1.1 Row parsing. (allow for multirow)
            for (row_header, cols_covered), row_idcs in row_headers.items():
                content = nondup_table[row_idcs, max(cols_covered)+1:]
                content = [x for x in content.reshape(-1).tolist() if x != '']
                questions.extend([
                    f"Parse the row starting with '{row_header}', and separate entries by ', '. Ignore whitespaces. If it covers multiple rows, go from left to right, and top to bottom."
                ])
                answers.extend([
                    ", ".join(content)
                ])
                
            # 1.2 Column parsing.
            for (col_header, cols_covered), col_idcs in column_headers.items():
                content = nondup_table[max(cols_covered)+1:, col_idcs]
                content = [x for x in content.T.reshape(-1).tolist() if x != '']
                questions.extend([
                    f"Parse the column starting with '{col_header}', and separate entries by ', '. If it covers multiple rows, go from top to bottom, and left to right. Ignore whitespaces."
                ])
                answers.extend([
                    ", ".join(content)
                ])

            # 1.3 Index parsing
            for row in range(max_rows):
                content = [x for x in nondup_table[row, :] if x != '']
                questions.extend([
                    f"Parse row {row+1}, and separate entries by ', '. Ignore whitespaces."
                ])
                answers.extend([
                    ", ".join(content)
                ])
            for col in range(max_cols):
                content = [x for x in nondup_table[:, col] if x != '']
                questions.extend([
                    f"Parse column {col+1}, and separate entries by ', ';but ignore whitespaces."
                ])
                answers.extend([
                    ", ".join(content)
                ])
            
            # 1.4 Exact content retrieval
            for row in range(max_rows):
                for col in range(max_cols):
                    if table[row, col] != '':
                        questions.extend([
                            f"Extract the content for row {row+1} and column {col+1}."
                        ])
                        answers.extend([table[row, col]])
            
            # 1.5 Content retrieval by name
            for (row_header, cols_covered), row_idcs in row_headers.items():
                for (col_header, cols_covered), col_idcs in column_headers.items():
                    content = nondup_table[np.ix_(row_idcs, col_idcs)]
                    content = [x for x in content.reshape(-1).tolist() if x != '']
                    questions.extend([
                        f"Parse the content for row starting with '{row_header}' and column starting with '{col_header}', and separate entries by ', '. If it covers multiple rows and columns, go from left to right, and top to bottom."
                    ])
                    answers.extend([
                        ", ".join(content)
                    ])

            # 1.6 Parse the entire table.
            questions.append("Parse the entire table, left to right and top to bottom. Separate entries by ',', and ignore whitespaces.")
            content = []
            for row in range(max_rows):
                for col in range(max_cols):
                    if nondup_table[row, col] != "":
                        content.append(nondup_table[row, col])
            answers.append(", ".join(content))

            if args.q_per_table > 0:
                sampled_idcs = set(sorted(np.random.choice(
                    np.arange(len(questions)),
                    np.clip(args.q_per_table, 0, len(questions)),
                    replace=False
                )))
                questions = [x for i, x in enumerate(questions) if i in sampled_idcs]
                answers = [x for i, x in enumerate(answers) if i in sampled_idcs]

            for q, a in zip(questions, answers):
                if a != '':
                    human_q = f"{q}<image>" if random.choice([0, 1]) else f"<image>{q}"
                    processed.append({
                        'id': uid,
                        'image': image_path,
                        'conversations': [
                            {"from": "human", "value": human_q},
                            {"from": "gpt", "value": a}
                        ]
                    })
                    uid += 1

    if args.q_per_table == -1:
        TOTAL_VALID = 8432821
        assert len(processed) == TOTAL_VALID, f"Incorrect number of processed question-answer pairs. Have {len(processed)}, need {TOTAL_VALID}."

    CHUNK_SIZE = 1_000_000
    n_chunks = len(processed) // 1_000_000 + 1
    for i in range(n_chunks):        
        with open(f"{OUTPUT_DIR}/preprocessed_{i}.jsonl", "w") as f:
            for entry in tqdm(processed[i * CHUNK_SIZE : (i+1) * CHUNK_SIZE], desc=f"Writing to jsonl-file ({i+1}/{n_chunks})", leave=False):
                f.write(json.dumps(entry) + "\n")

    ###
    print("Cleaning up...")
    shutil.move(image_folder, OUTPUT_DIR)
    os.remove(f"{OUTPUT_DIR}/FinTabNet.c-PDF_Annotations.tar.gz")
    os.remove(f"{OUTPUT_DIR}/FinTabNet.c-Structure.tar.gz")
    shutil.rmtree(f"{OUTPUT_DIR}/FinTabNet.c-PDF_Annotations")
    shutil.rmtree(f"{OUTPUT_DIR}/FinTabNet.c-Structure")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--q_per_table', type=int, default=-1)
    args = parser.parse_args()
    main(args)