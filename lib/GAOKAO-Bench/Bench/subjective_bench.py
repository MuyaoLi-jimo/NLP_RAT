import sys
import os
parent_path = os.path.dirname(sys.path[0])
if parent_path not in sys.path:
    sys.path.append(parent_path)

from Models.openai_gpt4 import OpenaiAPI
from bench_function import get_api_key, export_distribute_json, export_union_json
import os
import json
import time
import argparse


if __name__ == "__main__":

    ### An example of using OpenAI GPT-4 model to generate the json file for the benchmarking of the model
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_api_key', type=str)
    args = parser.parse_args()

    openai_api_key = args.openai_api_key
    model_name = "gpt-4"
    model_api = OpenaiAPI([openai_api_key], model_name=model_name)

    with open("Sub_Prompt.json", "r") as f:
        data = json.load(f)['examples']
        f.close()

    for i in range(len(data)):
        directory = "../Data/Subjective_Questions"

        keyword = data[i]['keyword']
        question_type = data[i]['type']
        zero_shot_prompt_text = data[i]['prefix_prompt']
        print(keyword)
        print(question_type)

        export_distribute_json(
            model_api, 
            model_name, 
            directory, 
            keyword, 
            zero_shot_prompt_text, 
            question_type, 
            parallel_num=1, 
        )

        export_union_json(
            directory, 
            model_name, 
            keyword,
            zero_shot_prompt_text,
            question_type
        )
