from typing import Optional, Tuple, List
import os
import torch
from clip.model import CLIP
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
from args import args_define
from pathlib import Path
from torchvision.transforms import Compose, CenterCrop, ToTensor, Normalize, Resize
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as FT
import PIL

if torch.cuda.is_available():
    device = torch.device("cuda")
    dtype = torch.float16
else:
    device = torch.device("cpu")
    dtype = torch.float32

import sys
if sys.argv[0].endswith('app.py'):
    tag = "utils.py"
    print(f"[{tag}] Running app.py")
    print(f"[{tag}] Replacing args to experiment args to ensure correct reproducibility")
    
    import argparse
    args = argparse.ArgumentParser().parse_args()
    print(f"[{tag}] args before: {args}")
    exp_args_path = Path(args.output_dir) / "args.json"
    if not exp_args_path.exists():
        raise FileNotFoundError(f"[{tag}] Cannot find args.json at {exp_args_path}")

    with open(exp_args_path, "r") as f:
        exp_args = json.load(f)
    
    args = argparse.Namespace(**exp_args)
    print(f"[{tag}] args after: {args}")
else:
    print("[utils.py] NOT running app.py")
    args = args_define.args

@torch.no_grad()
def extract_image_features(dataset: Dataset, clip_model: CLIP, batch_size: Optional[int] = 32,
                           num_workers: Optional[int] = 16, dress_type = None) -> Tuple[torch.Tensor, List[str]]:
    """
    Extracts image features from a dataset using a CLIP model.
    """
    # Create data loader
    loader = DataLoader(dataset=dataset, batch_size=batch_size,
                        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)

    index_features = []
    index_names = []
    try:
        print(f"extracting image features {dataset.__class__.__name__} - {dataset.split}")
    except Exception as e:
        pass

    # Extract features
    for batch in tqdm(loader):
        images = batch.get('image')
        names = batch.get('image_name')
        if images is None:
            images = batch.get('reference_image')
        if names is None:
            names = batch.get('reference_name')

        images = images.to(device)
        with torch.no_grad():
            batch_features = clip_model.encode_image(images)
            index_features.append(batch_features.cpu())
            index_names.extend(names)

    index_features = torch.vstack(index_features)
    if dress_type is None:
        np.save(f'feature/{args.dataset}/{args.model_type}/index_names.npy', index_names)
        torch.save(index_features, f'feature/{args.dataset}/{args.model_type}/index_features.pt')
    else:
        dir_path = f'feature/{args.dataset}/{dress_type}/{args.model_type}'
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        index_names_path = f'feature/{args.dataset}/{dress_type}/index_names.npy'
        index_features_path = f'{dir_path}/index_features.pt'
        np.save(index_names_path, index_names)
        torch.save(index_features, index_features_path)
    return index_features, index_names



PROJECT_ROOT = Path(__file__).absolute().parents[1].absolute()

class TargetPad:
    """
    If an image aspect ratio is above a target ratio, pad the image to match such target ratio.
    For more details see Baldrati et al. 'Effective conditioned and composed image retrieval combining clip-based features.' Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (2022).
    """

    def __init__(self, target_ratio: float, size: int):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        """
        self.size = size
        self.target_ratio = target_ratio

    def __call__(self, image: PIL.Image.Image) -> PIL.Image.Image:
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return FT.pad(image, padding, 0, 'constant')


def targetpad_transform(target_ratio: float, dim: int) -> torch.Tensor:
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim),
        Resize(dim, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(dim),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def collate_fn(batch):
    '''
    function which discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    '''
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def sample_dataset(dataset_json_path: str, n_samples: int):
    import json
    import random
    with open(dataset_json_path, 'r') as f:
        dataset = json.load(f)
    sampled_dataset = random.sample(dataset, n_samples)
    return sampled_dataset

if __name__ == "__main__":
    import json
    n_samples = 200
    save_dir = Path("FashionIQ_multi_opt_gpt35_5_p1") / "captions"
    for fn in os.listdir('FashionIQ'):
        if fn.endswith('.json'):
            sampled = sample_dataset(Path('FashionIQ') / fn, n_samples)

            # change cap.dress.val.json to cap.dress.val.1.json
            fn_save = fn.split('.')
            fn_save[-1] = '1.json'
            fn_save = '.'.join(fn_save)
            with open(save_dir / fn_save, 'w') as f:
                json.dump(sampled, f, indent=4)
