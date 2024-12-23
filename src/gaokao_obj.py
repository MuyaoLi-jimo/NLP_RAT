"""
the entrance of the system for gaokao
"""
import src.Method.Method as Method
from src.dataset.DataHelper import DatasetLoader
from src.evaluate.evaluate import evaluate
from src.utils import mp_utils,file_utils
from src.Method.rag import RAG_SYSTEM
from pathlib import Path, PosixPath
from functools import partial
import os
from rich import console


TEMPLATE = {
    "2010-2022_Math_II_MCQs-single_choice":{
        "cot_prompt": ["请你做一道数学选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：{}",["question"]],
    },
    "2010-2022_Math_I_MCQs-single_choice":{
        "cot_prompt": ["请你做一道数学选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：{}",["question"]],
    },
    "2010-2022_History_MCQs-single_choice":{
        "system_prompt": "你是一个非常优秀的高中生，正在作答一张历史试卷",
        "cot_prompt": ["请你做一道历史选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：{}",["question"]],
    },
    "2010-2022_Geography_MCQs-multi_question_choice": {
        "system_prompt": "你是一个非常优秀的高中生，正在作答一张地理试卷",
        "cot_prompt": ["请你做一道地理选择题，其中包含两到三个小题。\n请你一步一步思考。每一题你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：（1）【答案】 A <eoa>\n（2）【答案】 B <eoa>\n请你严格按照上述格式作答。\n{}",["question"]],
    },
    "2010-2022_Political_Science_MCQs-single_choice":{
        "cot_prompt": ["请你做一道政治选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：{}",["question"]],
    },
    "2010-2022_Physics_MCQs-multi_choice":{
        "system_prompt": "你是一个非常优秀的高中生，正在作答一张物理试卷",
        "cot_prompt": ["请你做一道物理选择题。\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出所有符合题意的答案，并写在【答案】和<eoa>之间。\n例如：【答案】 AB <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】... <eoa>\n请你严格按照上述格式作答。\n{}",["question"]],
    },
    "2010-2022_Chemistry_MCQs-single_choice": {
        "system_prompt": "你是一个非常优秀的高中生，正在作答一张化学试卷",
        "cot_prompt": ["请你做一道化学选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：{}",["question"]],
    },
    "2010-2022_Biology_MCQs-single_choice":{
        "system_prompt": "你是一个非常优秀的高中生，正在作答一场生物考试",
        "cot_prompt": ["请你做一道生物选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：{}",["question"]],
    },
    "2010-2013_English_MCQs-single_choice": {
        "cot_prompt": ["请你做一道英语选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：{}",["question"]],
    },
    "2010-2022_Chinese_Modern_Lit-multi_question_choice": {
        "cot_prompt": ["请你做一道语文阅读理解题，其中包含三个小题。\n请你一步一步思考。每一题你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：（1）【答案】 A <eoa>\n（2）【答案】 B <eoa>\n请你严格按照上述格式作答。\n{}",["question"]],
    },
    "2010-2022_Chinese_Lang_and_Usage_MCQs-multi_question_choice": {
        "cot_prompt": ["请你做一道语文选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n（1）【解析】 ... <eoe>\n【答案】 ... <eoa>\n（2）【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。如果不止一道题，请分别作答\n题目如下：{}",["question"]],
    },
    "2010-2022_English_Fill_in_Blanks-multi_question_choice": {
        "cot_prompt": ["请你做一道英语完形填空题,其中包含二十个小题。\n请你一步一步思考。每一题你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：（1）【答案】 A <eoa>\n（2）【答案】 B <eoa>\n请你严格按照上述格式作答。\n{}",["question"]],
    },
    "2010-2022_English_Reading_Comp-multi_question_choice": {
        "cot_prompt": ["请你做一道英语阅读理解题，其中包含三到五个小题。\n请你一步一步思考。每一题你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：（1）【答案】 A <eoa>\n（2）【答案】 B <eoa>\n请你严格按照上述格式作答。\n{}",["question"]],
    },
    "2012-2022_English_Cloze_Test-five_out_of_seven": {
        "cot_prompt": ["请回答下面的问题，将符合题意的五个选项的字母写在【答案】和<eoa>之间，例如“【答案】 A B C D E <eoa>\n请严格按照上述格式作答。\n{}",["question"]]
    },
}

RAG_SYSTEM_MAP = {
    "2010-2022_Chemistry_MCQs-single_choice":"chemistry",
}

def get_query(inputs:dict)->str:
    return inputs["question"]

def gaokao_obj_run(subset_name:str,method:str,model_name="DeepSeek-V2.5",log_fold:str = Path("logs"),dl:DatasetLoader = DatasetLoader()):
    dataset = dl.get_dataset("gaokao_obj")
    test_dataset = dataset[subset_name]["test"]
    store_fold_path=log_fold/"gaokao_obj"/method/f"{subset_name}.jsonl"
    store_fold_path.parent.mkdir(parents=True,exist_ok=True)
    command = {
        "temperature":0.7,
        "max_tokens":1024,
    }
    if model_name == "DeepSeek-V2.5":
        command.update({
            "api_key":os.environ["DEEPSEEK_API_KEY"],
            "api_base":"https://api.deepseek.com/v1",
            "model_id":"deepseek-chat",
        })
    # prepare for run
    wrapper = None
    if method == "cot":
        wrapper = partial(Method.plain,system_prompt=TEMPLATE[subset_name]["system_prompt"],
                          command=command,
                          input_template=TEMPLATE[subset_name]["cot_prompt"][0],input_template_keys=TEMPLATE[subset_name]["cot_prompt"][1])
    elif method == "rag":
        
        wrapper = partial(Method.rag,system_prompt=TEMPLATE[subset_name]["system_prompt"],
                                     command=command,
                                     input_template=TEMPLATE[subset_name]["cot_prompt"][0],input_template_keys=TEMPLATE[subset_name]["cot_prompt"][1],
                                     get_query_fn=get_query,retriever_name=RAG_SYSTEM_MAP[subset_name])
    else:
        raise ValueError(f"unknown method: {method}")
    
    # run
    mp_utils.get_multiple_response(wrapper,[test_dataset[idx] for idx in range(test_dataset.num_rows)],batch_size=20,store_fold_path=store_fold_path,slow=True)
    return 
        
def gaokao_obj_test(subset_name:str,method:str,model_name="DeepSeek-V2.5",log_fold:PosixPath=Path("logs"),dl:DatasetLoader = DatasetLoader()):
    dataset = dl.get_dataset("gaokao_obj")
    test_dataset = dataset[subset_name]["test"]
    data_path = log_fold/"gaokao_obj"/method/f"{subset_name}.jsonl"
    file_answers = file_utils.load_jsonl(data_path)
    # map to the format the datasetloader need
    answers = {d["id"]:d["answer"] for d in file_answers if d.get("success","none")=="none"}
    response = evaluate("gaokao_obj",answers,sub_dataset_name=subset_name,method="accuracy",dataset=dataset)
    total_number = len(test_dataset)
    model_accuracy = 0
    model_score = 0
    total_score = sum(test_dataset["score"])
    for answer in file_answers:
        uuid = answer["id"]
        if uuid in response:
            answer.update(response[uuid])
        model_accuracy +=  answer.get("success",0)
        model_score +=  answer.get("score",0)
    file_utils.dump_jsonl(file_answers,data_path)
    console.Console().log(f"total number: {total_number}, run number: {len(file_answers)}")
    console.Console().log(f"model accuracy: {model_accuracy/len(file_answers)*100:.2f}%")
    console.Console().log(f"model score: {model_score}, total score: {total_score}")

if __name__ =="__main__":
    dl = DatasetLoader()
    sub_tasks = [
                    "2010-2022_Chemistry_MCQs-single_choice"
                 ]
    for sub_task in sub_tasks:
        #gaokao_obj_run(sub_task,method="rag",model_name="DeepSeek-V2.5",dl=dl)
        gaokao_obj_test(sub_task,method="rag",model_name="DeepSeek-V2.5",dl=dl)
