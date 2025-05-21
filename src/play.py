from datasets import CIRRDataset, CIRCODataset, FashionIQDataset
from utils import extract_image_features, device, collate_fn, PROJECT_ROOT, targetpad_transform

preprocess = targetpad_transform(1.25, 224)
relative_val_dataset = FashionIQDataset(
    "FashionIQ_multi_opt_gpt35_5", 'val', ["dress"], 'relative', preprocess)

for data in relative_val_dataset:
    import pdb; pdb.set_trace()
