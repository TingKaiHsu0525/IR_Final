import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import json
from args import args_define
from typing import List, Tuple, Dict

import clip
import open_clip
import numpy as np
import torch
import torch.nn.functional as F
from clip.model import CLIP
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import CIRRDataset, CIRCODataset, FashionIQDataset
from utils import extract_image_features, device, collate_fn, PROJECT_ROOT, targetpad_transform
from utils import extract_text_features

import matplotlib.pyplot as plt
import PIL.Image
from pathlib import Path
import math

name2model = {
    'SEIZE-B':'ViT-B-32',
    'SEIZE-L':'ViT-L-14',
    'SEIZE-H':'ViT-H-14',
    'SEIZE-g':'ViT-g-14',
    'SEIZE-G':'ViT-bigG-14',
    'SEIZE-CoCa-B':'coca_ViT-B-32',
    'SEIZE-CoCa-L':'coca_ViT-L-14'
}

pretrained = {
    'ViT-B-32':'openai',
    'ViT-L-14':'openai', # For fair comparison, previous work used opanai's CLIP instead of open_clip
    'ViT-H-14':'laion2b_s32b_b79k', # Models larger than ViT-H only on open_clip, using laion2b uniformly
    'ViT-g-14':'laion2b_s34b_b88k',
    'ViT-bigG-14':'laion2b_s39b_b160k',
    'coca_ViT-B-32':'mscoco_finetuned_laion2b_s13b_b90k', # 'laion2b_s13b_b90k'
    'coca_ViT-L-14':'mscoco_finetuned_laion2b_s13b_b90k'  # 'laion2b_s13b_b90k'
}


@torch.no_grad()
def cirr_generate_test_submission_file(dataset_path: str, clip_model_name: str, preprocess: callable, submission_name: str) -> None:
    """
    Generate the test submission file for the CIRR dataset given the pseudo tokens
    """

    # Load the CLIP model
    clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, device=device, pretrained=pretrained[clip_model_name])
    clip_model = clip_model.float().eval().requires_grad_(False)

    # Compute the index features
    classic_test_dataset = CIRRDataset(dataset_path, 'test1', 'classic', preprocess)

    if os.path.exists(f'feature/{args.dataset}/{args.model_type}/index_features.pt'):
        index_features = torch.load(f'feature/{args.dataset}/{args.model_type}/index_features.pt')
        index_names = np.load(f'feature/{args.dataset}/index_names.npy')
        index_names = index_names.tolist()
    else:
        index_features, index_names = extract_image_features(classic_test_dataset, clip_model)

    relative_test_dataset = CIRRDataset(dataset_path, 'test1', 'relative', preprocess)

    # Get the predictions dicts
    pairid_to_retrieved_images, pairid_to_group_retrieved_images = \
        cirr_generate_test_dicts(relative_test_dataset, clip_model, index_features, index_names)

    submission = {
        'version': 'rc2',
        'metric': 'recall'
    }
    group_submission = {
        'version': 'rc2',
        'metric': 'recall_subset'
    }

    submission.update(pairid_to_retrieved_images)
    group_submission.update(pairid_to_group_retrieved_images)

    submissions_folder_path = PROJECT_ROOT / 'data' / "test_submissions" / 'cirr'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    with open(submissions_folder_path / f"{args.model_type}_{args.gpt_version}_{submission_name}.json", 'w+') as file:
        json.dump(submission, file, sort_keys=True)

    with open(submissions_folder_path / f"subset_{args.model_type}_{args.gpt_version}_{submission_name}.json", 'w+') as file:
        json.dump(group_submission, file, sort_keys=True)


def cirr_generate_test_dicts(relative_test_dataset: CIRRDataset, clip_model: CLIP, index_features: torch.Tensor,
                             index_names: List[str]) \
        -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Generate the test submission dicts for the CIRR dataset given the pseudo tokens
    """

    feat_dataset_path = f'feature/{args.dataset}'
    if os.path.exists(f'{feat_dataset_path}/{args.model_type}/{args.gpt_version}_predicted_features.pt'):
        reference_names = np.load(f'{feat_dataset_path}/reference_names.npy')
        pairs_id = np.load(f'{feat_dataset_path}/pairs_id.npy')
        group_members = np.load(f'{feat_dataset_path}/group_members.npy')
        predicted_features = torch.load(f'{feat_dataset_path}/{args.model_type}/{args.gpt_version}_predicted_features.pt')
        reference_names = reference_names.tolist()
        pairs_id = pairs_id.tolist()
        group_members = group_members.tolist()

    else:
        predicted_features, reference_names, pairs_id, group_members = \
            cirr_generate_test_predictions(clip_model, relative_test_dataset)      
        np.save(f'{feat_dataset_path}/reference_names.npy', reference_names)
        np.save(f'{feat_dataset_path}/pairs_id.npy', pairs_id)
        np.save(f'{feat_dataset_path}/group_members.npy', group_members)
        torch.save(predicted_features, f'{feat_dataset_path}/{args.model_type}/{args.gpt_version}_predicted_features.pt')

    if args.use_momentum_strategy:
        if os.path.exists(f'{feat_dataset_path}/{args.model_type}/blip_predicted_features.pt'):
            blip_predicted_features = torch.load(f'{feat_dataset_path}/{args.model_type}/blip_predicted_features.pt')
        else:
            blip_predicted_features, _, _, _ = \
                cirr_generate_test_predictions(clip_model, relative_test_dataset, True)
            
            torch.save(blip_predicted_features, f'{feat_dataset_path}/{args.model_type}/blip_predicted_features.pt')
    
    print(f"Compute CIRR prediction dicts")

    # Normalize the index features
    index_features = index_features.to(device)
    index_features = F.normalize(index_features, dim=-1).float()

    similarity_after = predicted_features @ index_features.T
    similarity_before = blip_predicted_features @ index_features.T

    diff_pos = similarity_after - similarity_before

    diff_pos[diff_pos < 0] = 0

    diff_neg = similarity_after - similarity_before

    diff_neg[diff_neg > 0] = 0

    similarity = similarity_after + args.neg_factor * diff_neg + args.pos_factor * diff_pos

    # similarity = similarity_after + args.momentum_factor * diff_neg + 0.3 * diff_pos

    # sorted_indices_before = torch.topk(similarity_before, dim=-1, k=similarity_before.shape[-1]).indices
    # sorted_indices_after = torch.topk(similarity_after, dim=-1, k=similarity_after.shape[-1]).indices

    # sorted_idx_before = torch.topk(sorted_indices_before, dim=-1, k=similarity_before.shape[-1], largest=False).indices
    # sorted_idx_after = torch.topk(sorted_indices_after, dim=-1, k=similarity_after.shape[-1], largest=False).indices

    # diff_neg = sorted_idx_before - sorted_idx_after
    # diff_neg[diff_neg > 0] = 0 

    # similarity = similarity_after + args.momentum_factor * diff_neg

    # for i in range(similarity_before.shape[0]):
    #     similarity[i][indexs[i]] = -1


    # Compute the distances and sort the results
    distances = 1 - similarity
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Delete the reference image from the results
    reference_mask = torch.tensor(
        sorted_index_names != np.repeat(np.array(reference_names), len(index_names)).reshape(len(sorted_index_names),
                                                                                             -1))
    sorted_index_names = sorted_index_names[reference_mask].reshape(sorted_index_names.shape[0],
                                                                    sorted_index_names.shape[1] - 1)
    # Compute the subset predictions
    group_members = np.array(group_members)
    group_mask = (sorted_index_names[..., None] == group_members[:, None, :]).sum(-1).astype(bool)
    sorted_group_names = sorted_index_names[group_mask].reshape(sorted_index_names.shape[0], -1)

    # Generate prediction dicts
    pairid_to_retrieved_images = {str(int(pair_id)): prediction[:50].tolist() for (pair_id, prediction) in
                                  zip(pairs_id, sorted_index_names)}
    pairid_to_group_retrieved_images = {str(int(pair_id)): prediction[:3].tolist() for (pair_id, prediction) in
                                        zip(pairs_id, sorted_group_names)}

    return pairid_to_retrieved_images, pairid_to_group_retrieved_images


def cirr_generate_test_predictions(clip_model: CLIP, relative_test_dataset: CIRRDataset, use_momentum_strategy=False) -> \
        Tuple[torch.Tensor, List[str], List[str], List[List[str]]]:
    """
    Generate the test prediction features for the CIRR dataset given the pseudo tokens
    """

    # Create the test dataloader
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32, num_workers=16,
                                      pin_memory=False)

    predicted_features_list = []
    reference_names_list = []
    pair_id_list = []
    group_members_list = []
    tokenizer = open_clip.get_tokenizer(name2model[args.model_type])

    # Compute the predictions
    for batch in tqdm(relative_test_loader):
        reference_names = batch['reference_name']
        pairs_id = batch['pair_id']
        relative_captions = batch['relative_caption']
        group_members = batch['group_members']
        multi_caption = batch['multi_{}'.format(args.caption_type)]
        multi_gpt_caption = batch['multi_gpt_{}'.format(args.caption_type)]

        group_members = np.array(group_members).T.tolist()

        # input_captions = [
        #     f"a photo of $ that {rel_caption}" for rel_caption in relative_captions]
        if use_momentum_strategy:
            input_captions = multi_caption
        else:
            input_captions = multi_gpt_caption
    
        text_features_list = []
        for cap in input_captions:
            tokenized_input_captions = tokenizer(cap, context_length=77).to(device)
            text_features = clip_model.encode_text(tokenized_input_captions)
            text_features_list.append(text_features)
        text_features_list = torch.stack(text_features_list)
        text_features = torch.mean(text_features_list, dim=0)

        predicted_features = F.normalize(text_features)
        predicted_features_list.append(predicted_features)
        reference_names_list.extend(reference_names)
        pair_id_list.extend(pairs_id)
        group_members_list.extend(group_members)

    predicted_features = torch.vstack(predicted_features_list)

    return predicted_features, reference_names_list, pair_id_list, group_members_list


@torch.no_grad()
def circo_generate_test_submission_file(dataset_path: str, clip_model_name: str, preprocess: callable,
                                        submission_name: str) -> None:
    """
    Generate the test submission file for the CIRCO dataset given the pseudo tokens
    """

    # Load the CLIP model
    clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, device=device, pretrained=pretrained[clip_model_name])
    clip_model = clip_model.float().eval().requires_grad_(False)

    # Compute the index features
    classic_test_dataset = CIRCODataset(dataset_path, 'test', 'classic', preprocess)

    if os.path.exists(f'feature/{args.dataset}/{args.model_type}/index_features.pt'):
        index_features = torch.load(f'feature/{args.dataset}/{args.model_type}/index_features.pt')
        index_names = np.load(f'feature/{args.dataset}/index_names.npy')
        index_names = index_names.tolist()
    else:
        index_features, index_names = extract_image_features(classic_test_dataset, clip_model)

    relative_test_dataset = CIRCODataset(dataset_path, 'test', 'relative', preprocess)

    # Get the predictions dict
    queryid_to_retrieved_images = circo_generate_test_dict(relative_test_dataset, clip_model, index_features,
                                                           index_names)

    submissions_folder_path = PROJECT_ROOT / 'data' / "test_submissions" / 'circo'
    submissions_folder_path.mkdir(exist_ok=True, parents=True)

    with open(submissions_folder_path / f"{args.model_type}_{args.gpt_version}_{submission_name}.json", 'w+') as file:
        json.dump(queryid_to_retrieved_images, file, sort_keys=True)


def circo_generate_test_predictions(clip_model: CLIP, relative_test_dataset: CIRCODataset, use_momentum_strategy=False) -> [torch.Tensor, List[List[str]]]:
    """
    Generate the test prediction features for the CIRCO dataset given the pseudo tokens
    """

    # Create the test dataloader
    # yzy num_workers
    relative_test_loader = DataLoader(dataset=relative_test_dataset, batch_size=32, num_workers=16,
                                      pin_memory=False, collate_fn=collate_fn, shuffle=False)

    predicted_features_list = []
    query_ids_list = []
    tokenizer = open_clip.get_tokenizer(name2model[args.model_type])

    # Compute the predictions
    for batch in tqdm(relative_test_loader):
        reference_names = batch['reference_name']
        relative_captions = batch['relative_caption']
        query_ids = batch['query_id']
        blip2_caption = batch['blip2_caption_{}'.format(args.caption_type)]
        gpt_caption = batch['gpt_caption_{}'.format(args.caption_type)]
        multi_caption = batch['multi_{}'.format(args.caption_type)]
        multi_gpt_caption = batch['multi_gpt_{}'.format(args.caption_type)]

        if use_momentum_strategy:
            input_captions = multi_caption
        else:
            input_captions = multi_gpt_caption

        text_features_list = []
        for cap in input_captions:
            tokenized_input_captions = tokenizer(cap, context_length=77).to(device)
            text_features = clip_model.encode_text(tokenized_input_captions)
            text_features_list.append(text_features)
        text_features_list = torch.stack(text_features_list)
        text_features = torch.mean(text_features_list, dim=0)
        predicted_features = F.normalize(text_features)

        predicted_features_list.append(predicted_features)
        query_ids_list.extend(query_ids)

    predicted_features = torch.vstack(predicted_features_list)
    return predicted_features, query_ids_list


def circo_generate_test_dict(relative_test_dataset: CIRCODataset, clip_model: CLIP, index_features: torch.Tensor,
                             index_names: List[str]) \
        -> Dict[str, List[str]]:
    """
    Generate the test submission dicts for the CIRCO dataset given the pseudo tokens
    """

    # Get the predicted features
    feat_dataset_path = f'feature/{args.dataset}'
    if os.path.exists(f'{feat_dataset_path}/{args.model_type}/{args.gpt_version}_predicted_features.pt'):
        predicted_features = torch.load(f'{feat_dataset_path}/{args.model_type}/{args.gpt_version}_predicted_features.pt')
        query_ids = np.load(f'{feat_dataset_path}/query_ids.npy')
    else:
        predicted_features, query_ids = circo_generate_test_predictions(clip_model, relative_test_dataset)
        np.save(f'{feat_dataset_path}/query_ids.npy', query_ids)
        torch.save(predicted_features, f'{feat_dataset_path}/{args.model_type}/{args.gpt_version}_predicted_features.pt')

    # Normalize the features
    index_features = index_features.float().to(device)
    index_features = F.normalize(index_features, dim=-1)

    if args.use_momentum_strategy:
        if os.path.exists(f'{feat_dataset_path}/{args.model_type}/blip_predicted_features.pt'):
            blip_predicted_features = torch.load(f'{feat_dataset_path}/{args.model_type}/blip_predicted_features.pt')
        else:
            blip_predicted_features, _ = circo_generate_test_predictions(clip_model, relative_test_dataset, True)
            torch.save(blip_predicted_features, f'{feat_dataset_path}/{args.model_type}/blip_predicted_features.pt')
        
        ref_names_list = np.load(f'{feat_dataset_path}/ref_names.npy')
        ref_names_list = ref_names_list.tolist()
        index_name_dict = {name: index for index, name in enumerate(index_names)}
        indexs = [index_name_dict[name] for name in ref_names_list]

        similarity_after = predicted_features @ index_features.T
        similarity_before = blip_predicted_features @ index_features.T

        diff_pos = similarity_after - similarity_before

        diff_pos[diff_pos < 0] = 0

        diff_neg = similarity_after - similarity_before

        diff_neg[diff_neg > 0] = 0

        similarity = similarity_after + args.neg_factor * diff_neg + args.pos_factor * diff_pos

        # similarity = similarity_after + args.momentum_factor * diff_neg + 0.3 * diff_pos

        # sorted_indices_before = torch.topk(similarity_before, dim=-1, k=similarity_before.shape[-1]).indices
        # sorted_indices_after = torch.topk(similarity_after, dim=-1, k=similarity_after.shape[-1]).indices

        # sorted_idx_before = torch.topk(sorted_indices_before, dim=-1, k=similarity_before.shape[-1], largest=False).indices
        # sorted_idx_after = torch.topk(sorted_indices_after, dim=-1, k=similarity_after.shape[-1], largest=False).indices

        # diff_neg = sorted_idx_before - sorted_idx_after
        # diff_neg[diff_neg > 0] = 0 

        # similarity = similarity_after + args.momentum_factor * diff_neg

        for i in range(similarity_before.shape[0]):
            similarity[i][indexs[i]] = -1

        # forward_steps = torch.zeros_like(similarity_after)
        # for i in range(sorted_indices_before.shape[0]):
        #     rank_blip = sorted_indices_before[i].numpy()
        #     rank_gpt = sorted_indices_after[i].numpy()
        #     idxs = np.arange(len(rank_blip))

        #     # 转换第一个向量为字典
        #     first_dict = {id: index for index, id in enumerate(rank_blip)}

        #     # 遍历第二个向量
        #     for idx, id in enumerate(rank_gpt):
        #         forward_steps[i][id] = first_dict[id] - idx
            
        # similarity = similarity_after + args.momentum_factor * forward_steps


    # Compute the similarity
    else:
        similarity = predicted_features @ index_features.T
    sorted_indices = torch.topk(similarity, dim=-1, k=50).indices.cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # Generate prediction dicts
    queryid_to_retrieved_images = {query_id: query_sorted_names[:50].tolist() for
                                   (query_id, query_sorted_names) in zip(query_ids, sorted_index_names)}

    return queryid_to_retrieved_images


@torch.no_grad()
def fiq_val_retrieval(dataset_path: str, dress_type: str, clip_model_name: str, ref_names_list: List[str], preprocess: callable) -> Dict[str, float]:
    """
    Compute the retrieval metrics on the FashionIQ validation set given the pseudo tokens and the reference names
    """
    # Load the model
    clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name, device=device, pretrained=pretrained[clip_model_name])
    clip_model = clip_model.float().eval().requires_grad_(False)

    # Extract the index features
    classic_val_dataset = FashionIQDataset(dataset_path, 'val', [dress_type], 'classic', preprocess)
    text_val_dataset = FashionIQDataset(dataset_path, 'val', [dress_type], 'text', preprocess)
    
    if os.path.exists(f'feature/{args.dataset}/{dress_type}/{args.model_type}/index_features.pt'):
        index_features = torch.load(f'feature/{args.dataset}/{dress_type}/{args.model_type}/index_features.pt')
        index_names = np.load(f'feature/{args.dataset}/{dress_type}/index_names.npy')
        index_names = index_names.tolist()
        index_text_features = torch.load(f'feature/{args.dataset}/{dress_type}/{args.model_type}/index_text_features.pt')
    else:
        index_features, index_names = extract_image_features(classic_val_dataset, clip_model, dress_type=dress_type)
        index_text_features, index_names = extract_text_features(text_val_dataset, clip_model, dress_type=dress_type)
    
    # Define the relative dataset
    relative_val_dataset = FashionIQDataset(dataset_path, 'val', [dress_type], 'relative', preprocess)

    return fiq_compute_val_metrics(relative_val_dataset, clip_model, index_features, index_text_features, index_names, ref_names_list,
                                   use_cache=True)


@torch.no_grad()
def fiq_compute_val_metrics(relative_val_dataset: FashionIQDataset, clip_model: CLIP, index_features: torch.Tensor, index_text_features: torch.Tensor,
                            index_names: List[str], ref_names_list: List[str], use_cache=False) \
        -> Dict[str, float]:
    """
    Compute the retrieval metrics on the FashionIQ validation set given the dataset, pseudo tokens and the reference names
    """

    dress_type = relative_val_dataset.dress_types[0]
    # Generate the predicted features
    feat_dataset_dress_path = f'feature/{args.dataset}/{dress_type}'
    if use_cache:
        if os.path.exists(f'{feat_dataset_dress_path}/{args.model_type}/{args.gpt_version}_predicted_features.pt'):
            target_names = np.load(f'{feat_dataset_dress_path}/target_names.npy')
            predicted_features = torch.load(f'{feat_dataset_dress_path}/{args.model_type}/{args.gpt_version}_predicted_features.pt')
            original_features = torch.load(f'{feat_dataset_dress_path}/{args.model_type}/original_features.pt')
            target_names = target_names.tolist()
        else:
            predicted_features, original_features, target_names = fiq_generate_val_predictions(clip_model, relative_val_dataset, ref_names_list
                                                                            )
            np.save(f'{feat_dataset_dress_path}/target_names.npy', target_names)
            torch.save(predicted_features, f'{feat_dataset_dress_path}/{args.model_type}/{args.gpt_version}_predicted_features.pt')
            torch.save(original_features, f'{feat_dataset_dress_path}/{args.model_type}/original_features.pt')
    else:
        print("Recalculating fiq_generate_val_predictions...")
        predicted_features, original_features, target_names = fiq_generate_val_predictions(
            clip_model, relative_val_dataset, ref_names_list)

    if args.use_momentum_strategy:
        if use_cache:
            if os.path.exists(f'{feat_dataset_dress_path}/{args.model_type}/blip_predicted_features.pt'):
                blip_predicted_features = torch.load(f'{feat_dataset_dress_path}/{args.model_type}/blip_predicted_features.pt')
            else:
                blip_predicted_features , *_ = \
                    fiq_generate_val_predictions(clip_model, relative_val_dataset, ref_names_list, args.use_momentum_strategy)
                
                torch.save(blip_predicted_features, f'{feat_dataset_dress_path}/{args.model_type}/blip_predicted_features.pt')
        else:
            blip_predicted_features , *_ = \
                fiq_generate_val_predictions(clip_model, relative_val_dataset, ref_names_list, args.use_momentum_strategy)

    # Move the features to the device
    index_features = index_features.to(device)
    predicted_features = predicted_features.to(device)
    original_features = original_features.to(device)
    index_text_features = index_text_features.to(device)

    # Normalize the features
    index_features = F.normalize(index_features.float(), dim=1)
    predicted_features = F.normalize(predicted_features.float(), dim=1)
    original_features = F.normalize(original_features.float(), dim=1)
    index_text_features = F.normalize(index_text_features.float(), dim=1)

    index_name_dict = {name: index for index, name in enumerate(index_names)}
    indexs = [index_name_dict[name] if name in index_name_dict else -1 for name in ref_names_list]


    similarity_after = predicted_features @ index_features.T
    similarity_before = blip_predicted_features @ index_features.T
    similarity_original = original_features @ index_features.T
    text_similarity = predicted_features @ index_text_features.T

    diff_pos = similarity_after - similarity_before

    diff_pos[diff_pos < 0] = 0

    diff_neg = similarity_after - similarity_before

    diff_neg[diff_neg > 0] = 0
    
    # original 
    # similarity = similarity_after + args.neg_factor * diff_neg + args.pos_factor * diff_pos
    
    # Modify: similarity
    similarity = args.beta*torch.sigmoid(text_similarity) + args.alpha*torch.sigmoid(similarity_original) + similarity_after + args.neg_factor * diff_neg + args.pos_factor * diff_pos
    
    for i in range(similarity_before.shape[0]):
        if indexs[i] != -1:
            similarity[i][indexs[i]] = -1

    print(f"Similarity range: min={similarity.min().item()}, max={similarity.max().item()}")
    # Compute the distances
    distances = 1 - similarity
    print(f"Distances range: min={distances.min().item()}, max={distances.max().item()}")
    sorted_indices = torch.argsort(distances, dim=-1).cpu()
    sorted_index_names = np.array(index_names)[sorted_indices]

    # save distances and sorted_index_names
    rin_dir = os.path.join(args.output_dir, "retrieved_index_names")
    np.save(os.path.join(rin_dir, f"{dress_type}_sorted_index_names.npy"), sorted_index_names)
    np.save(os.path.join(rin_dir, f"{dress_type}_distances.npy"), distances.cpu().numpy())

    # Check if the target names are in the top 10 and top 50
    labels = torch.tensor(
        sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
    assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())

    # Compute the metrics
    recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
    recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
    recall_at200 = (torch.sum(labels[:, :200]) / len(labels)).item() * 100

    return {'fiq_recall_at10': recall_at10,
            'fiq_recall_at50': recall_at50,
            'fiq_recall_at200': recall_at200}


@torch.no_grad()
def fiq_generate_val_predictions(clip_model: CLIP, relative_val_dataset: FashionIQDataset, ref_names_list: List[str],
                                 use_momentum_strategy=False) -> Tuple[torch.Tensor, List[str]]:
    """
    Generates features predictions for the validation set of Fashion IQ.
    """

    # Create data loader
    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=32, num_workers=16,
                                     pin_memory=False, collate_fn=collate_fn, shuffle=False)

    predicted_features_list = []
    predicted_features_list1 = []
    target_names_list = []
    tokenizer = open_clip.get_tokenizer(name2model[args.model_type])

    # Compute features
    for batch in tqdm(relative_val_loader):
        reference_names = batch['reference_name']
        target_names = batch['target_name']
        relative_captions = batch['relative_captions']
        multi_caption = batch['multi_{}'.format(args.caption_type)]
        multi_gpt_caption = batch['multi_gpt_{}'.format(args.caption_type)]
        original_captions = batch['captions']
        # flattened_captions: list = np.array(relative_captions).T.flatten().tolist()
        # input_captions = [
        #     f"{flattened_captions[i].strip('.?, ')} and {flattened_captions[i + 1].strip('.?, ')}" for
        #     i in range(0, len(flattened_captions), 2)]
        # input_captions_reversed = [
        #     f"{flattened_captions[i + 1].strip('.?, ')} and {flattened_captions[i].strip('.?, ')}" for
        #     i in range(0, len(flattened_captions), 2)]

        input_caption_original = original_captions

        if use_momentum_strategy:
            input_captions = multi_caption
        else:
            input_captions = multi_gpt_caption

        # input_captions = [
        #     f"a photo of $ that {in_cap}" for in_cap in input_captions]
        # tokenized_input_captions = tokenizer(input_captions, context_length=77).to(device)
        # text_features = encode_with_pseudo_tokens(clip_model, tokenized_input_captions, batch_tokens)

        # input_captions_reversed = [
        #     f"a photo of $ that {in_cap}" for in_cap in input_captions_reversed]
        # tokenized_input_captions_reversed = tokenizer(input_captions_reversed, context_length=77).to(device)
        # text_features_reversed = encode_with_pseudo_tokens(clip_model, tokenized_input_captions_reversed,
        #                                                    batch_tokens)


        text_features_list = []
        original_features_list = []
        for cap in input_captions:
            tokenized_input_captions = tokenizer(cap, context_length=77).to(device)
            text_features = clip_model.encode_text(tokenized_input_captions)
            text_features_list.append(text_features)

        for cap in input_caption_original:
            tokenized_input_captions = tokenizer(cap, context_length=77).to(device)
            text_features = clip_model.encode_text(tokenized_input_captions)
            original_features_list.append(text_features)

        text_features_list = torch.stack(text_features_list)
        text_features = torch.mean(text_features_list, dim=0)

        original_features_list = torch.stack(original_features_list)
        original_features = torch.mean(original_features_list, dim=0)

        predicted_features = F.normalize(text_features)
        predicted_features1 = F.normalize(original_features)
        # predicted_features = F.normalize((F.normalize(text_features) + F.normalize(text_features_reversed)) / 2)
        # predicted_features = F.normalize((text_features + text_features_reversed) / 2)

        predicted_features_list.append(predicted_features)
        predicted_features_list1.append(predicted_features1)
        target_names_list.extend(target_names)

    predicted_features = torch.vstack(predicted_features_list)
    predicted_features1 = torch.vstack(predicted_features_list1)
    return predicted_features, predicted_features1, target_names_list

def show_topk_grid(dataset_path: Path,
                   sorted_index_names: np.ndarray,
                   distances: np.ndarray,
                   top_k: int = 50,
                   ncols: int = 10,
                   img_ext: str = ".png"):
    """
    Display the top_k retrieved images in a grid, ordered by increasing distance.
    
    Args:
        dataset_path: Path to a FashionIQ root (must contain an 'images/' subfolder).
        sorted_index_names: 1D array of all index_names sorted by distance ascending.
        distances: 1D tensor of distances.
        top_k: how many of the top items to show.
        ncols: how many columns in the grid.
        img_ext: file extension of your images (e.g. ".png" or ".jpg").
    """
    topk_names = sorted_index_names[:top_k]

    sorted_indices = np.argsort(distances)[:top_k]
    topk_dists = distances[sorted_indices]
    
    nrows = math.ceil(top_k / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))
    axes = axes.flatten()
    
    for i, (name, dist) in enumerate(zip(topk_names, topk_dists)):
        img_path = dataset_path / "images" / (str(name) + img_ext)
        img = PIL.Image.open(img_path)
        axes[i].imshow(img)
        axes[i].set_title(f"{dist:.4f}", fontsize=8)
        axes[i].axis("off")
    
    # hide any leftover axes
    for j in range(top_k, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.show()


def visualize(output_dir, item_idx, dress_type):
    with open(os.path.join(output_dir, "args.json"), 'r') as f:
        exp_args = json.load(f)
    
    if exp_args["model_type"] in ['SEIZE-B', 'SEIZE-L', 'SEIZE-H', 'SEIZE-g', 'SEIZE-G', 'SEIZE-CoCa-B', 'SEIZE-CoCa-L']:
        preprocess = targetpad_transform(1.25, 224)
    else:
        raise ValueError("Model type not supported")
    
    dataset_path = Path(exp_args["dataset_path"])

    # Note that FashionIQDataset still depends on args
    # To ensure that FashionIQDataset gives the right data
    # We need to make sure that args match exp_args
    arg_mismatch_msg = "The arguments of standard input do not match \
the arguments specified in the experiment checkpoint \
(the args.json file in the experiement checkpoint directory).\n\
Please make sure they match, or the visualization results may be incorrect."
    for k, v in vars(args).items():
        if not (k in exp_args and exp_args[k] == v):
            arg_mismatch_msg += f"\nAt argument {k}: {v} (standard input) != {exp_args[k]} (experiment)."
            raise ValueError(
                arg_mismatch_msg
            )

    # Note: FashionIQDataset depends on args
    relative_val_dataset = FashionIQDataset(
        exp_args["dataset_path"], 'val', [dress_type], 'relative', preprocess)
    
    item = relative_val_dataset[item_idx]

    print(f"dataset item:")
    for k, v in item.items():
        if "image" in k:
            continue
        print(f"{k}: {v}")

    # print(f"Input editing caption: {item['relative_captions']}")
    # print(f"multi_opt (caption for reference image): {item['multi_opt']}")
    # print(f"multi_gpt (caption for reference image): {item['multi_gpt']}")

    # Show reference image and ground truth target image (not preprocessed)
    ref_img = PIL.Image.open(
        dataset_path / "images" / (item["reference_name"] + ".png"))
    target_img = PIL.Image.open(
        dataset_path / "images" / (item["target_name"] + ".png"))
    
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.imshow(np.array(ref_img)); ax1.axis("off"); ax1.set_title("Reference Image")
    # ax2.imshow(np.array(target_img)); ax2.axis("off"); ax2.set_title("Target Image")
    
    #plt.ion() 
    # plt.show()

    # Show retrieved images
    distances = np.load(
        Path(output_dir) / "retrieved_index_names" / f"{dress_type}_distances.npy")
    sorted_index_names = np.load(
        Path(output_dir) / "retrieved_index_names" / f"{dress_type}_sorted_index_names.npy")

    # Check if sorted_index_names and distances have 1-1 correspondence with
    # data in relative_val_dataset
    assert sorted_index_names.shape[0] == len(relative_val_dataset)
    assert distances.shape[0] == len(relative_val_dataset)

    # Find where target_name is at
    target_idx = np.asarray(
        sorted_index_names[item_idx] == item["target_name"]
    ).nonzero()[0][0]
    print(f"Target found at index {target_idx} in sorted_index_names.")

    top_k = 3
    ncols = 2 + top_k
    fig, axes = plt.subplots(1, ncols, figsize=(ncols * 2, 2))
    axes = axes.flatten()

    axes[0].imshow(np.array(ref_img))
    axes[0].axis("off")
    axes[0].set_title("Input Image")

    topk_names = sorted_index_names[item_idx, :top_k]
    sorted_indices = np.argsort(distances[item_idx])[:top_k]
    topk_dists = distances[item_idx, sorted_indices]

    for i, (name, dist) in enumerate(zip(topk_names, topk_dists)):
        img_path = dataset_path / "images" / (str(name) + ".png")
        img = PIL.Image.open(img_path)
        axes[i+1].imshow(np.array(img))
        axes[i+1].axis("off")
        axes[i+1].set_title(f"Top {i+1} (distance={dist:.4f})")

    axes[1+top_k].imshow(np.array(target_img))
    axes[1+top_k].axis("off")
    axes[1+top_k].set_title(f"Ground Truth Image (top {target_idx})")

    plt.tight_layout()
    plt.show()

    # show_topk_grid(
    #     dataset_path,
    #     sorted_index_names[item_idx],
    #     distances[item_idx]
    # )



args = args_define.args
print(f"Args:\n{args}")

def main():
    if args.model_type in ['SEIZE-B', 'SEIZE-L', 'SEIZE-H', 'SEIZE-g', 'SEIZE-G', 'SEIZE-CoCa-B', 'SEIZE-CoCa-L']:
        clip_model_name = name2model[args.model_type]
        preprocess = targetpad_transform(1.25, 224)
    else:
        raise ValueError("Model type not supported")

    preprocess = targetpad_transform(1.25, 224)

    folder_path = f'feature/{args.dataset}/{args.model_type}/'
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    # set up experiment output directory
    outputs_root = os.path.join(args.dataset_path, "outputs")
    if not os.path.exists(outputs_root):
        os.makedirs(outputs_root)
    # name output by datetime
    from datetime import datetime
    output_dir = os.path.join(outputs_root, datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(output_dir, exist_ok=False)
    args.output_dir = output_dir

    # Set up output_dir, which includes 
    # saving args for record keeping and reproducibility
    with open(os.path.join(output_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    os.makedirs(os.path.join(output_dir, "retrieved_index_names"))

    if args.dataset == 'cirr':
        cirr_generate_test_submission_file(args.dataset_path, clip_model_name, preprocess, args.submission_name)
    elif args.dataset == 'circo':
        circo_generate_test_submission_file(args.dataset_path, clip_model_name, preprocess, args.submission_name)
    elif args.dataset.lower() == 'fashioniq':
        recalls_at10 = []
        recalls_at50 = []
        recalls_at200 = []
        for dress_type in ['shirt', 'dress', 'toptee']:
        #for dress_type in ['dress']: # one type

            # get refernce_name list
            with open(f'FashionIQ_multi_opt_gpt35_5/captions/cap.{dress_type}.val.json', 'r') as f:
                data = json.load(f)

            # 提取所有的 candidate 欄位值
            ref_names_lists = [item['candidate'] for item in data]

            fiq_metrics = fiq_val_retrieval(args.dataset_path, dress_type, clip_model_name, ref_names_lists, preprocess)
            recalls_at10.append(fiq_metrics['fiq_recall_at10'])
            recalls_at50.append(fiq_metrics['fiq_recall_at50'])
            recalls_at200.append(fiq_metrics['fiq_recall_at200'])

            for k, v in fiq_metrics.items():
                print(f"{dress_type}_{k} = {v:.2f}")
            print("\n")

        print(f"average_fiq_recall_at10 = {np.mean(recalls_at10):.2f}")
        print(f"average_fiq_recall_at50 = {np.mean(recalls_at50):.2f}")    
        print(f"average_fiq_recall_at200 = {np.mean(recalls_at200):.2f}")
    else:
        raise ValueError("Dataset not supported")


if __name__ == '__main__':
    main()
    
    # output_dir = "FashionIQ_cap_num_15_split1/outputs/20250604_154326"
    # visualize(output_dir, item_idx=5, dress_type='shirt')
