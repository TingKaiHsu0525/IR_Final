import json
import time
import traceback

from tqdm import tqdm
# import openai
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

SPLIT = 'val'
BLIP2_MODEL = 'opt' # 'opt' or 't5'
MULTI_CAPTION = True

# openai.api_key = ""

# modift no use openai
# model_name = "meta-llama/Llama-2-7b-chat-hf"  # 也可以選更小的模型
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")


def generate_response(system_prompt, user_prompt):
    # 將 system_prompt 與 user_prompt 合併為一個 prompt
    prompt = f"<s>[INST] <<SYS>>{system_prompt}<</SYS>> {user_prompt} [/INST]"
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    output_ids = model.generate(input_ids, max_new_tokens=256)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # 從 output 擷取出 "Edited Description:" 之後的內容
    if "Edited Description:" in response:
        ret = response.split("Edited Description:")[-1].strip()
    else:
        ret = response.strip()
    return ret


# input_json = '{}_blip2_{}.json'.format(SPLIT, BLIP2_MODEL)
dataset_path = Path('FashionIQ')
# input_json = 'FashionIQ/captions/cap.dress.val.json'
# dataset_path = Path('CIRR')

# original code
# for DRESS in ['dress', 'shirt', 'toptee']:

#     # original caption path
#     # with open(dataset_path / 'captions' / f'cap.{DRESS}.{SPLIT}.json', "r") as f:
#     #     annotations = json.load(f)

#     # modified caption path
#     with open(dataset_path  / f'cap.{DRESS}.{SPLIT}.json', "r") as f:
#         annotations = json.load(f)

#     for ans in tqdm(annotations):
#         rel_caps = ans["captions"]
#         rel_cap = rel_caps[0] + rel_caps[1]
#         if MULTI_CAPTION:
#             blip2_caption = ans["multi_caption_{}".format(BLIP2_MODEL)]
#         else:
#             if BLIP2_MODEL == 'none':
#                 blip2_caption = ans["shared_concept"]
#             elif BLIP2_MODEL == 'opt':
#                 blip2_caption = ans["blip2_caption"]
#             else:
#                 blip2_caption = ans["blip2_caption_{}".format(BLIP2_MODEL)]

#         sys_prompt = "I have an image. Given an instruction to edit the image, carefully generate a description of the edited image."

#         if MULTI_CAPTION:
#             multi_gpt = []
#             for cap in blip2_caption:
#                 usr_prompt = "I will put my image content beginning with \"Image Content:\". The instruction I provide will begin with \"Instruction:\". The edited description you generate should begin with \"Edited Description:\". You just generate one edited description only begin with \"Edited Description:\". The edited description needs to be as simple as possible and only reflects image content. Just one line.\nA example:\nImage Content: a man adjusting a woman's tie.\nInstruction: has the woman and the man with the roles switched.\nEdited Description: a woman adjusting a man's tie.\n\nImage Content: {}\nInstruction: {}\nEdited Description:".format(cap, rel_cap)
#                 # print(prompt)
#                 while True:
#                     try:
#                         completion = openai.ChatCompletion.create(
#                             model="gpt-3.5-turbo",
#                             messages=[{"role": "system",
#                                         "content": sys_prompt},
#                                         {"role": "user", "content": usr_prompt}],
#                             timeout=10, request_timeout=10)
#                         ret = completion['choices'][0]["message"]["content"].strip('\n')
#                         multi_gpt.append(ret)
#                         break
#                     except:
#                         traceback.print_exc()
#                         time.sleep(3)
#             ans["multi_gpt-3.5_{}".format(BLIP2_MODEL)] = multi_gpt
            
#         else:
#             usr_prompt = "I will put my image content beginning with \"Image Content:\". The instruction I provide will begin with \"Instruction:\". The edited description you generate should begin with \"Edited Description:\". You just generate one edited description only begin with \"Edited Description:\". The edited description needs to be as simple as possible and only reflects image content. Just one line.\nA example:\nImage Content: a man adjusting a woman's tie.\nInstruction: has the woman and the man with the roles switched.\nEdited Description: a woman adjusting a man's tie.\n\nImage Content: {}\nInstruction: {}\nEdited Description:".format(blip2_caption, rel_cap)
#             # print(prompt)
#             while True:
#                 try:
#                     completion = openai.ChatCompletion.create(
#                         model="gpt-3.5-turbo",
#                         messages=[{"role": "system",
#                                     "content": sys_prompt},
#                                     {"role": "user", "content": usr_prompt}],
#                         timeout=10, request_timeout=10)
#                     ret = completion['choices'][0]["message"]["content"].strip('\n')
#                     if BLIP2_MODEL == 'opt':
#                         ans["gpt-3.5-turbo"] = ret
#                     else:
#                         ans["gpt-3.5-turbo_{}".format(BLIP2_MODEL)] = ret
#                     break
#                 except:
#                     traceback.print_exc()
#                     time.sleep(3)

#         # with open("CIRCO/annotations/gpt3.5-temp.json", "a") as f:
#         #     f.write(json.dumps(ans, indent=4) + '\n')

#     with open(dataset_path / f'new.cap.{DRESS}.{SPLIT}.json', "w") as f:
#         f.write(json.dumps(annotations, indent=4))


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

        sys_prompt = "I have an image. Given an instruction to edit the image, carefully generate a description of the edited image."

        if MULTI_CAPTION:
            multi_gpt = []
            for cap in blip2_caption:
                usr_prompt = (
                    "I will put my image content beginning with \"Image Content:\". "
                    "The instruction I provide will begin with \"Instruction:\". "
                    "The edited description you generate should begin with \"Edited Description:\". "
                    "You just generate one edited description only begin with \"Edited Description:\". "
                    "The edited description needs to be as simple as possible and only reflects image content. Just one line.\n"
                    "A example:\n"
                    "Image Content: a man adjusting a woman's tie.\n"
                    "Instruction: has the woman and the man with the roles switched.\n"
                    "Edited Description: a woman adjusting a man's tie.\n\n"
                    f"Image Content: {cap}\nInstruction: {rel_cap}\nEdited Description:"
                )
                ret = generate_response(sys_prompt, usr_prompt)
                multi_gpt.append(ret)
            ans[f"multi_llama2_{BLIP2_MODEL}"] = multi_gpt
        else:
            usr_prompt = (
                "I will put my image content beginning with \"Image Content:\". "
                "The instruction I provide will begin with \"Instruction:\". "
                "The edited description you generate should begin with \"Edited Description:\". "
                "You just generate one edited description only begin with \"Edited Description:\". "
                "The edited description needs to be as simple as possible and only reflects image content. Just one line.\n"
                "A example:\n"
                "Image Content: a man adjusting a woman's tie.\n"
                "Instruction: has the woman and the man with the roles switched.\n"
                "Edited Description: a woman adjusting a man's tie.\n\n"
                f"Image Content: {blip2_caption}\nInstruction: {rel_cap}\nEdited Description:"
            )
            ret = generate_response(sys_prompt, usr_prompt)
            ans[f"llama2_{BLIP2_MODEL}"] = ret

    with open(dataset_path / f'new.cap.{DRESS}.{SPLIT}.json', "w") as f:
        f.write(json.dumps(annotations, indent=4))
