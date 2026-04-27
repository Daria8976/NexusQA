import argparse
import random
import torch
import os
import json
from tqdm import tqdm
import re
from collections import defaultdict
import glob
import numpy as np

np.set_printoptions(precision=2)

from torch.utils.data import Dataset, DataLoader

from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

from gelm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from gelm.model.builder import load_pretrained_model
from gelm.utils import load_video_frames


class VideoQATestset(Dataset):

    def __init__(self, args, questions, tokenizer, image_processor, model_config):
        self.args = args
        self.questions = questions
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        qs_item = self.questions[idx]

        qs = qs_item['question']
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[self.args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        prompt = prompt.strip()
        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')

        image_tensor = torch.load(os.path.join(self.args.image_folder, qs_item['video_id'] + '.pth.tar'),
                                  map_location="cpu")

        return {
            'segment_id': qs_item['video_id'],
            'sample_id': qs_item['sample_id'],
            'question': qs_item['question'],
            'answer': qs_item['answer_complete'],
            "answer_complete": qs_item['answer_complete'],
            "answer_summary": qs_item['answer_summary'],
            "answer_evidence": qs_item['answer_evidence'],
            "evidence": qs_item['evidence'],
            'input_ids': input_ids,
            'image_tensor': image_tensor,
        }

def convert_to_json_serializable(data):
    """
    Recursively converts tensors in a nested structure to JSON serializable types.
    """
    if isinstance(data, torch.Tensor):
        # If it is a Tensor, convert according to the dimension
        if data.dim() == 0:
            return data.item()  # Convert 0-dimensional tensor to a number
        else:
            return data.tolist() # Convert multi-dimensional tensor to a list
    elif isinstance(data, list):
        # If it is a list, recursively convert each element in it
        return [convert_to_json_serializable(item) for item in data]
    elif isinstance(data, dict):
        # If it is a dictionary, recursively convert each value in it
        return {key: convert_to_json_serializable(value) for key, value in data.items()}
    else:
        # Other types (int, float, str) are returned directly
        return data

def interval_intersection(interval1, interval2):
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])
    return max(0, end - start + 1)

def IoU(pred, gt):
    """Compute intersection over union
    pred, gt: list of list of intervals ([[s1, e1], [s2, e2], ...])
    """
    pred = [pred] if isinstance(pred[0], int) else pred
    gt = [gt] if isinstance(gt[0], int) else gt
    intersection = sum(interval_intersection(p, g) for p in pred for g in gt)
    total_pred = sum(p[1] - p[0] + 1 for p in pred)
    total_gt = sum(g[1] - g[0] + 1 for g in gt)
    union = total_pred + total_gt - intersection

    return intersection / union if union != 0 else 0

def IoP(pred, gt):
    """Compute intersection over predicted intervals
    pred, gt: list of list of intervals ([[s1, e1], [s2, e2], ...])
    """
    pred = [pred] if isinstance(pred[0], int) else pred
    gt = [gt] if isinstance(gt[0], int) else gt
    intersection = sum(interval_intersection(p, g) for p in pred for g in gt)
    total_pred = sum(p[1] - p[0] + 1 for p in pred)

    return intersection / total_pred if total_pred != 0 else 0


def IoG(pred, gt):
    """Compute intersection over ground truth intervals
    pred, gt: list of list of intervals ([[s1, e1], [s2, e2], ...])
    """
    pred = [pred] if isinstance(pred[0], int) else pred
    gt = [gt] if isinstance(gt[0], int) else gt
    intersection = sum(interval_intersection(p, g) for p in pred for g in gt)
    total_gt = sum(g[1] - g[0] + 1 for g in gt)

    return intersection / total_gt if total_gt != 0 else 0


# DataLoader
def create_data_loader(questions, image_folder, tokenizer, image_processor, model_config, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = VideoQATestset(questions, image_folder, tokenizer, image_processor, model_config)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):

    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)
    assert not model.config.mm_use_im_start_end and not model.config.mm_use_im_patch_token, "not supported yet"

    if args.conv_mode is None:
        if "llama-2" in model_name.lower():
            conv_mode = "llava_llama_2"
        elif "tinyllama" in model.name.lower():
            conv_mode = "tiny_llama"
        elif "v1" in model_name.lower():
            conv_mode = "llava_v1"
        elif "mpt" in model_name.lower():
            conv_mode = "mpt"
        else:
            conv_mode = "llava_v0"
        args.conv_mode = conv_mode

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))  #[:1]
    data_loader = create_data_loader(args, questions, tokenizer, image_processor, model.config)

    results = []

    pbar = tqdm(data_loader, total=len(data_loader))
    for idx, sample in enumerate(pbar):

        input_ids = sample['input_ids']
        image_tensor = sample['image_tensor']

        stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style not in [
            SeparatorStyle.TWO, SeparatorStyle.TINY_LLAMA
        ] else conv_templates[args.conv_mode].sep2
        input_ids = input_ids.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            output_ids, saliency_proposals, similarity_proposals = model.generate_with_time(
                input_ids,
                images=image_tensor.to(dtype=torch.float32, device='cuda', non_blocking=True),
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=1024,
                use_cache=True,
                multi_hop=True,
                factor_saliency=0.7,
                factor_similarity=0.1,
            )

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        output_ids = output_ids[0, input_token_len:]

        outputs = tokenizer.batch_decode([output_ids], skip_special_tokens=False)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = re.sub(r'<\/?T\d*>', '', outputs)
        outputs = re.sub(r'\s+([.,!?])', r'\1', outputs)
        outputs = re.sub(r'\s+', ' ', outputs)
        outputs = outputs.strip()

        results.append({
            "sample_id": int(sample['sample_id']),
            "video_id": sample["segment_id"],
            "Q": sample['question'],
            "A": outputs,
            "answer_complete": sample['answer_complete'],
            "answer_summary": sample['answer_summary'],
            "answer_evidence": sample['answer_evidence'],
            "evidence": convert_to_json_serializable(sample["evidence"]),
            "sim_T": convert_to_json_serializable(similarity_proposals),
            "sal_T": convert_to_json_serializable(saliency_proposals),
        })

        pbar.update(1)
        pbar.set_postfix({"id": sample['sample_id'][0]})

    pbar.close()
    save_path = os.path.join(args.output_dir, "nature_v2.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--visual-data-type", type=str, default="video_frames")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--output-dir", type=str, default="./outputs")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)