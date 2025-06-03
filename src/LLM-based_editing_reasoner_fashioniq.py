import json
import time
import traceback

from tqdm import tqdm
# import openai
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
SPLIT = 'val'
BLIP2_MODEL = 'opt' # 'opt' or 't5'
MULTI_CAPTION = True

# openai.api_key = ""

# modift no use openai
model_name = "meta-llama/Llama-2-7b-chat-hf"  # 也可以選更小的模型
# model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name,
                            torch_dtype=torch.float16).to("cuda")
 


def generate_response(system_prompt, user_prompt):
    # 將 system_prompt 與 user_prompt 合併為一個 prompt
    prompt = f"<s>[INST] <<SYS>>{system_prompt}<</SYS>>\nNow, please rewrite the image description based on the following:\n{user_prompt} [/INST]"
    #print(prompt)
    # with torch.no_grad():
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(input_ids, max_new_tokens=256)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    #response = tokenizer.decode(output_ids[0][input_ids.shape[-1]:], skip_special_tokens=True)

    # response = re.sub(r"<</?SYS>>", "", response)
    # response = re.sub(r"\[/?INST\]", "", response)

    # print("----- RAW RESPONSE -----")
    # print(response)
    
    # 從 output 擷取出 "Edited Description:" 之後的內容
    if "Edited Description:" in response:
        # ret = response.split("Edited Description:")[-1].strip()
        ret = response.split("Edited Description:")[-1].strip().split("\n")[0]
    else:
        #ret = response.strip()
        ret = "[ERROR] " + response.strip()
    
    # print("----- FINAL RESPONSE -----")
    # print(ret)
    return ret

# input_json = '{}_blip2_{}.json'.format(SPLIT, BLIP2_MODEL)
dataset_path = Path('FashionIQ')
# input_json = 'FashionIQ/captions/cap.dress.val.json'
# dataset_path = Path('CIRR')


# modified code
for DRESS in ['dress', 'shirt', 'toptee']:
    with open(dataset_path / f'cap.{DRESS}.{SPLIT}.json', "r") as f:
        annotations = json.load(f)

    for ans in tqdm(annotations):
        rel_caps = ans["captions"]
        rel_cap = rel_caps[0] + rel_caps[1]
        
        if MULTI_CAPTION:
            blip2_caption = ans[f"multi_caption_{BLIP2_MODEL}"]
        else:
            if BLIP2_MODEL == 'none':
                blip2_caption = ans["shared_concept"]
            elif BLIP2_MODEL == 'opt':
                blip2_caption = ans["blip2_caption"]
            else:
                blip2_caption = ans[f"blip2_caption_{BLIP2_MODEL}"]
        
        #sys_prompt = "I have an image. Given an instruction to edit the image, carefully generate a description of the edited image."

        sys_prompt = (
            "You are a helpful assistant that rewrites image descriptions after an editing instruction.\n"
            "The format is:\n"
            "- Image Content: describes the original image\n"
            "- Instruction: describes what change is desired\n"
            "- Edited Description: a single, short sentence describing the edited image\n\n"
            "Example:\n"
            "Image Content: a man adjusting a woman's tie\n"
            "Instruction: has the woman and the man with the roles switched\n"
            "Edited Description: a woman adjusting a man's tie"
        )

        if MULTI_CAPTION:
            multi_gpt = []
            for cap in blip2_caption:
                # print(cap)
                # usr_prompt = (
                #     "I will put my image content beginning with \"Image Content:\". "
                #     "The instruction I provide will begin with \"Instruction:\". "
                #     "The edited description you generate should begin with \"Edited Description:\". "
                #     "You just generate one edited description only begin with \"Edited Description:\". "
                #     "The edited description needs to be as simple as possible and only reflects image content. Just one line.\n"
                #     "A example:\n"
                #     "Image Content: a man adjusting a woman's tie.\n"
                #     "Instruction: has the woman and the man with the roles switched.\n"
                #     "Edited Description: a woman adjusting a man's tie.\n\n"
                #     f"Image Content: {cap}\nInstruction: {rel_cap}\nEdited Description:"
                # )
                #usr_prompt = "I will put my image content beginning with \"Image Content:\". The instruction I provide will begin with \"Instruction:\". The edited description you generate should begin with \"Edited Description:\". You just generate one edited description only begin with \"Edited Description:\". The edited description needs to be as simple as possible and only reflects image content. Just one line.\nA example:\nImage Content: a man adjusting a woman's tie.\nInstruction: has the woman and the man with the roles switched.\nEdited Description: a woman adjusting a man's tie.\n\nImage Content: {}\nInstruction: {}\nEdited Description:".format(cap, rel_cap)
                usr_prompt = f"Image Content: {cap}\nInstruction: {rel_cap}"
                ret = generate_response(sys_prompt, usr_prompt)
                multi_gpt.append(ret)
            ans[f"multi_llama2_{BLIP2_MODEL}"] = multi_gpt
        else:
            # usr_prompt = (
            #     "I will put my image content beginning with \"Image Content:\". "
            #     "The instruction I provide will begin with \"Instruction:\". "
            #     "The edited description you generate should begin with \"Edited Description:\". "
            #     "You just generate one edited description only begin with \"Edited Description:\". "
            #     "The edited description needs to be as simple as possible and only reflects image content. Just one line.\n"
            #     "A example:\n"
            #     "Image Content: a man adjusting a woman's tie.\n"
            #     "Instruction: has the woman and the man with the roles switched.\n"
            #     "Edited Description: a woman adjusting a man's tie.\n\n"
            #     f"Image Content: {blip2_caption}\nInstruction: {rel_cap}\nEdited Description:"
            # )
            #usr_prompt = "I will put my image content beginning with \"Image Content:\". The instruction I provide will begin with \"Instruction:\". The edited description you generate should begin with \"Edited Description:\". You just generate one edited description only begin with \"Edited Description:\". The edited description needs to be as simple as possible and only reflects image content. Just one line.\nA example:\nImage Content: a man adjusting a woman's tie.\nInstruction: has the woman and the man with the roles switched.\nEdited Description: a woman adjusting a man's tie.\n\nImage Content: {}\nInstruction: {}\nEdited Description:".format(blip2_caption, rel_cap)
            usr_prompt = f"Image Content: {blip2_caption}\nInstruction: {rel_cap}"
            ret = generate_response(sys_prompt, usr_prompt)
            ans[f"llama2_{BLIP2_MODEL}"] = ret

    with open(dataset_path / f'new.cap.{DRESS}.{SPLIT}.json', "w") as f:
        f.write(json.dumps(annotations, indent=4))