import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
from pathlib import Path
import torch
from tqdm import tqdm
import PIL.Image as Image
from lavis.models import load_model_and_preprocess

SPLIT = 'val'
# DRESS = 'toptee' # 'shirt' 'toptee'
BLIP2_MODEL = 'opt' # or 'opt'
MULTI_CAPTION = True
NUM_CAPTION = 5 # 15

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

if BLIP2_MODEL == 'opt':
    # original used opt6.7b
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_opt", model_type="caption_coco_opt6.7b", is_eval=True, device=device
    )

    # modifyd to use opt2.7b
    # model, vis_processors, _ = load_model_and_preprocess(
    # name="blip2_opt", model_type="caption_coco_opt2.7b", is_eval=True, device=device
    # )

else:
    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_t5", model_type="pretrain_flant5xxl", is_eval=True, device=device
    )
# model = model.float()
    
for DRESS in ['dress', 'shirt', 'toptee']:
    output_json = '{}_{}_multi.json'.format(SPLIT, BLIP2_MODEL) if MULTI_CAPTION else '{}_{}.json'.format(SPLIT, BLIP2_MODEL)
    dataset_path = Path('FashionIQ')

    with open(dataset_path / 'captions' / f'cap.{DRESS}.{SPLIT}.json') as f:
        annotations = json.load(f)

    for ans in tqdm(annotations):
        ref_img_name = ans["candidate"] + '.png'

        reference_img_path = dataset_path / 'images' / ref_img_name

        raw_image = Image.open(reference_img_path).convert('RGB')
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        if MULTI_CAPTION:
            caption = model.generate({"image": image}, use_nucleus_sampling=True, num_captions=NUM_CAPTION)
        else:
            caption = model.generate({"image": image})

        if MULTI_CAPTION:
            ans["multi_caption_{}".format(BLIP2_MODEL)] = caption
        else:
            ans["blip2_caption_{}".format(BLIP2_MODEL)] = caption[0]

        # new_annotations.append(ans)
        # with open("CIRCO/annotations/blip2_caption_t5.json", "a") as f:
        #     f.write(json.dumps(ans, indent=4) + '\n')

    with open(dataset_path / f'cap.{DRESS}.{SPLIT}.json', "w") as f:
        f.write(json.dumps(annotations, indent=4))
