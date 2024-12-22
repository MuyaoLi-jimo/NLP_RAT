"""
the entrance of the system for mbpp
"""
import src.Method.Method as Method
from src.dataset.DataHelper import DatasetLoader
from src.evaluate.evaluate import evaluate,mbpp_examiner
from src.utils import file_utils,mp_utils
from functools import partial
from pathlib import Path, PosixPath

import os


API_KEY = os.environ["DEEPSEEK_API_KEY"]

SYSTEM_PROMPT = "You are an expert in code writing"



TEMPLATE = {
    "input":["{}\n The name of the function you should write is given, please follow this namespace: {}.\n\nYou should answer like this: ######\n <your code>  \n\nWarning: You should follow the format I give to you, or you will answer wrong.\n",["prompt","def"]],
    "cot":["{}\n The name of the function you should write is given, please follow this namespace: {}.\n\nYou should answer like this: <your thoughs>\n######\n <your code>  \n\nWarning: You should follow the format I give to you, or you will answer wrong.\n",["prompt","def"]],
    "icl_input":["Question: {},  The name of the function: {}",["prompt","def"]],
    "output":["Answer: ######\n {}\n",["code"]],
    "reflex_input":["Your answer is wrong. Here is the return informations: {} Please rewrite it\n\nThe new code:"],
}

def mbpp_run(method:str,log_fold:PosixPath=Path(__file__).parent/"logs",dl:DatasetLoader = DatasetLoader()):
    """use llm to get the code  """
    dataset = dl.get_dataset("mbpp")
    test_dataset = dataset["test"]
    command = {
        "api_key":API_KEY,
        "api_base":"https://api.deepseek.com/v1",
        "model_id":"deepseek-chat",
        "temperature":0.0,
        "max_tokens":1024,
    }
    # prepare for run
    if method == "plain":
        wrapper = partial(Method.plain,system_prompt=SYSTEM_PROMPT,command=command,input_template=TEMPLATE["input"][0],input_template_keys=TEMPLATE["input"][1])
        
    elif method == "cot":
        wrapper = partial(Method.cot,system_prompt=SYSTEM_PROMPT,command=command,input_template=TEMPLATE["cot"][0],input_template_keys=TEMPLATE["cot"][1])
        
    elif method == "icl":
        wrapper = partial(Method.icl,system_prompt=SYSTEM_PROMPT,k=4,
                                            command=command,examples=[dataset["prompt"][idx] for idx in range(dataset["prompt"].num_rows)] ,
                                            input_template=TEMPLATE["icl_input"][0],input_template_keys=TEMPLATE["icl_input"][1],
                                            output_template=TEMPLATE["output"][0],output_template_keys=TEMPLATE["output"][1],
                                            )
    elif method == "reflexion":
        command["temperature"]=0.3
        wrapper = partial(Method.reflexion,system_prompt=SYSTEM_PROMPT,command=command,
                                                     input_template=TEMPLATE["input"][0],input_template_keys=TEMPLATE["input"][1],
                                                     reflex_input_template=TEMPLATE["reflex_input"][0],
                                                     interact_fc=mbpp_examiner,roll=1)
    else:
        raise ValueError(f"unknown method: {method}")
    mp_utils.get_multiple_response(wrapper,[test_dataset[idx] for idx in range(test_dataset.num_rows)],batch_size=20,store_fold_path=log_fold/f"mbpp-{method}.jsonl",slow=True)
    return
    
def mbpp_test(method:str,answers:list=None,data_fold:PosixPath=Path(__file__).parent/"logs",data_path:str=None):
    if answers is None:
        if data_path is None:
            data_path = data_fold / f"mbpp-{method}.jsonl"
        file_answers = file_utils.load_jsonl(data_path)
        # map to the format the datasetloader need
        
        answers = {d["id"]:d["answer"] for d in file_answers if d.get("success","none")=="none"}
    response = evaluate("mbpp",answers,method="accuracy")
    total_accuracy = 0
    for answer in file_answers:
        uuid = answer["id"]
        if uuid in response:
            answer["success"] = response[uuid]
        total_accuracy +=  answer.get("success",0)
    file_utils.dump_jsonl(file_answers,data_path)
    print(f"{method}: accuracy--{total_accuracy/len(file_answers)}")
    return
    
    
if __name__ =="__main__":
    
    methods = ["reflexion"]
    #for method in methods:
        #mbpp_run(method)
    for method in methods:
        mbpp_test(method)
    