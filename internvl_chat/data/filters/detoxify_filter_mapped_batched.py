import os
import json
import random
from tqdm import tqdm

from argparse import ArgumentParser
from argparse import Namespace

import torch
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
import transformers
import time

class JSONLConversationDataset(Dataset):
    def __init__(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx]
        text = ' '.join(str(d["value"]) for d in line["conversations"] if d is not None)
        return line, text

def collate_fn(batch):
    lines, texts = zip(*batch)
    return list(lines), list(texts)

### Copied a bunch of code from: https://github.com/unitaryai/detoxify/blob/master/detoxify/detoxify.py
DOWNLOAD_URL = "https://github.com/unitaryai/detoxify/releases/download/"
MODEL_URLS = {
    "original": DOWNLOAD_URL + "v0.1-alpha/toxic_original-c1212f89.ckpt",
    "unbiased": DOWNLOAD_URL + "v0.3-alpha/toxic_debiased-c7548aa0.ckpt",
    "multilingual": DOWNLOAD_URL + "v0.4-alpha/multilingual_debiased-0b549669.ckpt",
    "original-small": DOWNLOAD_URL + "v0.1.2/original-albert-0e1d6498.ckpt",
    "unbiased-small": DOWNLOAD_URL + "v0.1.2/unbiased-albert-c8519128.ckpt",
}
PRETRAINED_MODEL = None
MAX_CHUNKS = 10

def get_model_and_tokenizer(
    model_type, model_name, tokenizer_name, num_classes, state_dict, huggingface_config_path=None
):
    model_class = getattr(transformers, model_name)
    config = model_class.config_class.from_pretrained(model_type, num_labels=num_classes)
    model = model_class.from_pretrained(
        pretrained_model_name_or_path=None,
        config=huggingface_config_path or config,
        state_dict=state_dict,
        local_files_only=huggingface_config_path is not None,
    )
    tokenizer = getattr(transformers, tokenizer_name).from_pretrained(
        huggingface_config_path or model_type,
        local_files_only=huggingface_config_path is not None,
        # TODO: may be needed to let it work with Kaggle competition
        # model_max_length=512,
    )

    return model, tokenizer


def load_checkpoint(model_type="original", checkpoint=None, device="cpu", huggingface_config_path=None):
    if checkpoint is None:
        checkpoint_path = MODEL_URLS[model_type]
        loaded = torch.hub.load_state_dict_from_url(checkpoint_path, map_location=device)
    else:
        loaded = torch.load(checkpoint, map_location=device)
        if "config" not in loaded or "state_dict" not in loaded:
            raise ValueError(
                "Checkpoint needs to contain the config it was trained \
                    with as well as the state dict"
            )
    class_names = loaded["config"]["dataset"]["args"]["classes"]
    # standardise class names between models
    change_names = {
        "toxic": "toxicity",
        "identity_hate": "identity_attack",
        "severe_toxic": "severe_toxicity",
    }
    class_names = [change_names.get(cl, cl) for cl in class_names]
    model, tokenizer = get_model_and_tokenizer(
        **loaded["config"]["arch"]["args"],
        state_dict=loaded["state_dict"],
        huggingface_config_path=huggingface_config_path,
    )

    return model, tokenizer, class_names


def load_model(model_type, checkpoint=None):
    if checkpoint is None:
        model, _, _ = load_checkpoint(model_type=model_type)
    else:
        model, _, _ = load_checkpoint(checkpoint=checkpoint)
    return model


class Detoxify:
    """Detoxify
    Easily predict if a comment or list of comments is toxic.
    Can initialize 5 different model types from model type or checkpoint path:
        - original:
            model trained on data from the Jigsaw Toxic Comment
            Classification Challenge
        - unbiased:
            model trained on data from the Jigsaw Unintended Bias in
            Toxicity Classification Challenge
        - multilingual:
            model trained on data from the Jigsaw Multilingual
            Toxic Comment Classification Challenge
        - original-small:
            lightweight version of the original model
        - unbiased-small:
            lightweight version of the unbiased model
    Args:
        model_type(str): model type to be loaded, can be either original,
                         unbiased or multilingual
        checkpoint(str): checkpoint path, defaults to None
        device(str or torch.device): accepts any torch.device input or
                                     torch.device object, defaults to cpu
        huggingface_config_path: path to HF config and tokenizer files needed for offline model loading
    Returns:
        results(dict): dictionary of output scores for each class
    """

    def __init__(self, model_type="original", checkpoint=PRETRAINED_MODEL, device="cpu", huggingface_config_path=None):
        super().__init__()
        self.model, self.tokenizer, self.class_names = load_checkpoint(
            model_type=model_type,
            checkpoint=checkpoint,
            device=device,
            huggingface_config_path=huggingface_config_path,
        )
        self.class_indices = list(range(len(self.class_names)))
        self.device = device
        self.model.to(self.device)

        if torch.__version__ >= "2.0" and args.device == "cuda":
            try:
                self.model = torch.compile(self.model)
                print("Using compiled model!")
            except Exception as e:
                print(f"[warning] torch.compile failed: {e}, using non-compiled model!")

    #### 14/04 (Vishaal): We patch this function to ensure that 
    #### we can take chunks of 512 size so that we never have
    #### to do truncation
    #### note this doesn't do batching --- we never use it!!
    @torch.no_grad()
    def predict(self, text):

        self.model.eval()

        # get all input-ids for full sequence without truncation
        tokens = self.tokenizer(text, return_tensors="pt", truncation=False, padding=True).to(self.model.device)

        # get the input-ids and attention mask
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        # we will now start chunking in 512-size since that's the context len of the roberta model we are using
        chunk_size = 512
        scores_list = []

        # iterate over full sequence until we see all tokens in a chunk-size of 512
        for i in range(0, len(input_ids), chunk_size):
            chunk_input_ids = input_ids[i : i+chunk_size]
            chunk_attention_mask = attention_mask[i : i+chunk_size]

            # pad if chunk is less than chunk_size
            pad_len = chunk_size - chunk_input_ids.shape[0]
            if pad_len > 0:
                chunk_input_ids = torch.nn.functional.pad(chunk_input_ids, (0, pad_len), value=self.tokenizer.pad_token_id)
                chunk_attention_mask = torch.nn.functional.pad(chunk_attention_mask, (0, pad_len), value=0)

            # get the chunk to process
            chunk = {
                "input_ids": chunk_input_ids.unsqueeze(0).to(self.model.device),
                "attention_mask": chunk_attention_mask.unsqueeze(0).to(self.model.device),
            }

            # get current chunk scores
            with autocast(device_type='cuda', enabled=(self.device == "cuda")):
                out = self.model(**chunk)[0]
                scores = torch.sigmoid(out).cpu()
            scores_list.append(scores)

        # Aggregate scores across chunks with max since we only care about the chunk with the maximum scores
        # Note: this might introduce some weird artefacts when splitting up some text sequences but its a
        # a reasonable approximation to what we need!
        all_scores = torch.stack(scores_list).max(dim=0).values

        results = []
        # If the input is a single text, squeezing will remove the dimensionality from the tensor -
        # so `.tolist()` will return a number instead. Otherwise, we'll get the list of scores of that class.
        for i, cla in enumerate(self.class_names):
            results.append(all_scores[:, i].squeeze().tolist())

        # Based on Datacomp-clip, we only want to exclude the sample
        # if any of the category scores exceeds a threshold of 0.1.
        # see appendix E in https://arxiv.org/pdf/2304.14108.
        return max(results) <= 0.1


    #### 14/04 (Vishaal): this is the batched version of the
    #### above function, this is what we use!!
    @torch.no_grad()
    def predict_many_batched(self, texts, chunk_size=512):
        """
        Truly batched version of predict_many.
        Returns a list of booleans indicating if each text is non-toxic (True) or toxic (False).
        """

        self.model.eval()

        all_input_ids = []
        all_attention_masks = []
        mapping = []  # maps each chunk to the original text index

        # 1. Tokenize without truncation
        for text_idx, text in enumerate(texts):
            tokens = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=False,
                padding=False
            )
            input_ids = tokens["input_ids"].squeeze(0)
            attention_mask = torch.ones_like(input_ids)  # since we didn't use tokenizer's padding

            current_chunks = 0
            for i in range(0, len(input_ids), chunk_size):

                # only allow upto certain max chunk-size
                if current_chunks > MAX_CHUNKS:
                    break

                chunk_ids = input_ids[i : i + chunk_size]
                chunk_mask = attention_mask[i : i + chunk_size]

                pad_len = chunk_size - chunk_ids.shape[0]
                if pad_len > 0:
                    chunk_ids = torch.nn.functional.pad(chunk_ids, (0, pad_len), value=self.tokenizer.pad_token_id)
                    chunk_mask = torch.nn.functional.pad(chunk_mask, (0, pad_len), value=0)

                all_input_ids.append(chunk_ids)
                all_attention_masks.append(chunk_mask)
                mapping.append(text_idx)

                current_chunks += 1

        # 2. Stack all chunks
        input_ids_batch = torch.stack(all_input_ids).to(self.model.device)  # [N_chunks, 512]
        attention_mask_batch = torch.stack(all_attention_masks).to(self.model.device)

        # 3. Single forward pass

        with autocast(device_type='cuda', enabled=(self.device == "cuda")):
            logits = self.model(input_ids=input_ids_batch, attention_mask=attention_mask_batch).logits
            scores = torch.sigmoid(logits).cpu()  # [N_chunks, num_labels]

        # 4. Aggregate max score per text
        num_texts = len(texts)
        max_scores = [torch.full((len(self.class_names),), -float("inf")) for _ in range(num_texts)]

        for i, idx in enumerate(mapping):
            max_scores[idx] = torch.max(max_scores[idx], scores[i][self.class_indices])

        # 6. Apply threshold rule
        return [torch.all(score <= 0.1).item() for score in max_scores]


def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as file_ptr:
        data = [json.loads(line) for line in file_ptr]
    return data

def make_output_path(args: Namespace):
    assert isinstance(args.input_file, str) \
        and args.input_file.endswith(".jsonl") \
            and args.input_file.count(".jsonl") == 1
    return args.input_file.replace(".jsonl", f"_{args.filter_name}.jsonl")


def test(input_file: str, output_file: str, filter_name: str):
    """
    Use this function as a final "unit-test" for your script (may take some time if your .jsonl is large). 
    :param input_file: path to input jsonl file;  
    :param output_file: path to output jsonl file:
    """
    
    # load input and output .jsonl files
    input_data = load_jsonl(input_file) 
    output_data = load_jsonl(output_file)

    # no lines should be added or removed
    assert len(output_data) == len(input_data)

    # files should be row-wise identical except for the new key
    for row_in, row_out in zip(input_data, output_data):

        # make sure new key is there
        assert filter_name in row_out

        # make sure that new value added is binary
        assert row_out[filter_name] in [0, 1]

        # make sure other values did not change
        for k, v in row_in.items():
            assert row_out[k] == v, \
                f"Value at key {k} changed from {v} to {row_out[k]}"
            
    print("Test ran through with no assertion errors =)")

def main(args):

    # get time stats
    st = time.time()

    model_ = Detoxify('multilingual', device=args.device)
    output_file = make_output_path(args)

    dataset = JSONLConversationDataset(args.input_file)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=args.prefetch_factor,
    )

    tot_samps = 0
    tot_removed = 0

    with open(output_file, "w", encoding="utf-8") as out_ptr:
        for batch_lines, batch_texts in tqdm(dataloader, total=len(dataloader), ascii=True):
            preds = model_.predict_many_batched(batch_texts)
            for line, pred in zip(batch_lines, preds):
                tot_samps += 1
                if not pred:
                    tot_removed += 1
                line[args.filter_name] = 1 if pred else 0
                out_ptr.write(json.dumps(line) + "\n")

    print("Took {} seconds to write everything!".format(time.time()-st))
    print("Removed: {}, Total: {}, Percent removed: {}".format(tot_removed, tot_samps, 100. * tot_removed/tot_samps))

    return args.input_file, output_file


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--filter_name', type=str, default="detoxify")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--prefetch_factor', type=int, default=6)
    args = parser.parse_args()
    
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    # Assert that the TORCH_HOME env is set!
    assert "TORCH_HOME" in os.environ, "TORCH_HOME is not set, model will be downloaded!!"

    # filtering here
    input_file, output_file = main(args)

    # run a simple unit-test
    test(input_file, output_file, args.filter_name)
