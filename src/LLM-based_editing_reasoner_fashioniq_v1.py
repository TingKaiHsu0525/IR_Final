import json
import time
import traceback

from tqdm import tqdm
import openai
from pathlib import Path

#from transformers import AutoModelForCausalLM, AutoTokenizer
import os

SPLIT = 'val'
BLIP2_MODEL = 'opt' # 'opt' or 't5'
MULTI_CAPTION = True

openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# input_json = '{}_blip2_{}.json'.format(SPLIT, BLIP2_MODEL)
dataset_path = Path('FashionIQ')
# input_json = 'FashionIQ/captions/cap.dress.val.json'
# dataset_path = Path('CIRR')

#original code
for DRESS in ['dress', 'shirt', 'toptee']:

    # original caption path
    # with open(dataset_path / 'captions' / f'cap.{DRESS}.{SPLIT}.json', "r") as f:
    #     annotations = json.load(f)

    # modified caption path
    with open(dataset_path  / f'cap.{DRESS}.{SPLIT}.json', "r") as f:
        annotations = json.load(f)

    for ans in tqdm(annotations):
        rel_caps = ans["captions"]
        rel_cap = rel_caps[0] + rel_caps[1]
        if MULTI_CAPTION:
            blip2_caption = ans["multi_caption_{}".format(BLIP2_MODEL)]
        else:
            if BLIP2_MODEL == 'none':
                blip2_caption = ans["shared_concept"]
            elif BLIP2_MODEL == 'opt':
                blip2_caption = ans["blip2_caption"]
            else:
                blip2_caption = ans["blip2_caption_{}".format(BLIP2_MODEL)]

        sys_prompt = "I have an image. Given an instruction to edit the image, carefully generate a description of the edited image."

        if MULTI_CAPTION:
            multi_gpt = []
            for cap in blip2_caption:
                usr_prompt = "I will put my image content beginning with \"Image Content:\". The instruction I provide will begin with \"Instruction:\". The edited description you generate should begin with \"Edited Description:\". You just generate one edited description only begin with \"Edited Description:\". The edited description needs to be as simple as possible and only reflects image content. Just one line.\nA example:\nImage Content: a man adjusting a woman's tie.\nInstruction: has the woman and the man with the roles switched.\nEdited Description: a woman adjusting a man's tie.\nNow, please rewrite the image description based on the following:\nImage Content: {}\nInstruction: {}\nEdited Description:".format(cap, rel_cap)
                # print("[usr_prompt]")
                # print(usr_prompt)
                # print("--"*20)
                while True:
                    try:
                        completion = openai.ChatCompletion.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "system",
                                        "content": sys_prompt},
                                        {"role": "user", "content": usr_prompt}],
                            timeout=10, request_timeout=10)
                        ret = completion['choices'][0]["message"]["content"].strip('\n')
                        # print("[GPT 3.5 Response]")
                        # print(ret)
                        # print("--"*20)

                        # 提取 Edited Description: 後的內容
                        if "Edited Description:" in ret:
                            ret = ret.split("Edited Description:")[-1].strip()
                        # print("[GPT 3.5 Response]")
                        # print(ret)
                        # print("--"*20)
                        multi_gpt.append(ret)
                        break
                    except:
                        traceback.print_exc()
                        time.sleep(3)
            ans["multi_gpt-3.5_{}".format(BLIP2_MODEL)] = multi_gpt
            
        else:
            usr_prompt = "I will put my image content beginning with \"Image Content:\". The instruction I provide will begin with \"Instruction:\". The edited description you generate should begin with \"Edited Description:\". You just generate one edited description only begin with \"Edited Description:\". The edited description needs to be as simple as possible and only reflects image content. Just one line.\nA example:\nImage Content: a man adjusting a woman's tie.\nInstruction: has the woman and the man with the roles switched.\nEdited Description: a woman adjusting a man's tie.\n\nImage Content: {}\nInstruction: {}\nEdited Description:".format(blip2_caption, rel_cap)
            print("[usr_prompt]")
            print(usr_prompt)
            print("--"*20)
            while True:
                try:
                    completion = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "system",
                                    "content": sys_prompt},
                                    {"role": "user", "content": usr_prompt}],
                        timeout=10, request_timeout=10)
                    ret = completion['choices'][0]["message"]["content"].strip('\n')
                    # print("[GPT 3.5 Response]")
                    # print(ret)
                    # print("--"*20)
                    if BLIP2_MODEL == 'opt':
                        ans["gpt-3.5-turbo"] = ret
                    else:
                        ans["gpt-3.5-turbo_{}".format(BLIP2_MODEL)] = ret
                    break
                except:
                    traceback.print_exc()
                    time.sleep(3)

        # with open("CIRCO/annotations/gpt3.5-temp.json", "a") as f:
        #     f.write(json.dumps(ans, indent=4) + '\n')

    with open(dataset_path / f'new.cap.{DRESS}.{SPLIT}.json', "w") as f:
        f.write(json.dumps(annotations, indent=4))


# # modified code
# for DRESS in ['dress', 'shirt', 'toptee']:
#     with open(dataset_path / f'cap.{DRESS}.{SPLIT}.json', "r") as f:
#         annotations = json.load(f)

#     for ans in tqdm(annotations):
#         rel_caps = ans["captions"]
#         rel_cap = rel_caps[0] + rel_caps[1]
        
#         if MULTI_CAPTION:
#             blip2_caption = ans[f"multi_caption_{BLIP2_MODEL}"]
#         else:
#             if BLIP2_MODEL == 'none':
#                 blip2_caption = ans["shared_concept"]
#             elif BLIP2_MODEL == 'opt':
#                 blip2_caption = ans["blip2_caption"]
#             else:
#                 blip2_caption = ans[f"blip2_caption_{BLIP2_MODEL}"]

#         sys_prompt = "I have an image. Given an instruction to edit the image, carefully generate a description of the edited image."

#         if MULTI_CAPTION:
#             multi_gpt = []
#             for cap in blip2_caption:
#                 usr_prompt = (
#                     "I will put my image content beginning with \"Image Content:\". "
#                     "The instruction I provide will begin with \"Instruction:\". "
#                     "The edited description you generate should begin with \"Edited Description:\". "
#                     "You just generate one edited description only begin with \"Edited Description:\". "
#                     "The edited description needs to be as simple as possible and only reflects image content. Just one line.\n"
#                     "A example:\n"
#                     "Image Content: a man adjusting a woman's tie.\n"
#                     "Instruction: has the woman and the man with the roles switched.\n"
#                     "Edited Description: a woman adjusting a man's tie.\n\n"
#                     f"Image Content: {cap}\nInstruction: {rel_cap}\nEdited Description:"
#                 )
#                 ret = generate_response(sys_prompt, usr_prompt)
#                 multi_gpt.append(ret)
#             ans[f"multi_llama2_{BLIP2_MODEL}"] = multi_gpt
#         else:
#             usr_prompt = (
#                 "I will put my image content beginning with \"Image Content:\". "
#                 "The instruction I provide will begin with \"Instruction:\". "
#                 "The edited description you generate should begin with \"Edited Description:\". "
#                 "You just generate one edited description only begin with \"Edited Description:\". "
#                 "The edited description needs to be as simple as possible and only reflects image content. Just one line.\n"
#                 "A example:\n"
#                 "Image Content: a man adjusting a woman's tie.\n"
#                 "Instruction: has the woman and the man with the roles switched.\n"
#                 "Edited Description: a woman adjusting a man's tie.\n\n"
#                 f"Image Content: {blip2_caption}\nInstruction: {rel_cap}\nEdited Description:"
#             )
#             ret = generate_response(sys_prompt, usr_prompt)
#             ans[f"llama2_{BLIP2_MODEL}"] = ret

#     with open(dataset_path / f'new.cap.{DRESS}.{SPLIT}.json', "w") as f:
#         f.write(json.dumps(annotations, indent=4))
