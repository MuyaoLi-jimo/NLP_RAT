"""
the entrance of the system for gaokao
"""
import src.Method.Method as Method
from src.dataset.DataHelper import DatasetLoader
from src.evaluate.evaluate import evaluate
from src.utils import mp_utils,file_utils
from src.Method.rag import RAG_SYSTEM
from src.Method.rat_tools import llm_split,get_query_naive,rag_retrieve
from pathlib import Path, PosixPath
from functools import partial
import os
from rich import console
import argparse

REFINE_PROMPT_TEMPLATE = """
你现在正在进行一个思维链，你在刚才已经完成了前几步的思考与修改，
现在你准备继续这个思路往下作答，并且完成了一个初稿，
但是这个分析可能存在错误，因此我寻找了一部分资料来检验分析是否正确
我希望你参考检索到的内容来修订这一步的分析结果。
你需要检查回答是否正确。
如果你发现检索到的内容是垃圾信息，直接输出原始回答即可。
如果发现回答中有错误，请修改回答使其更好。
如果发现有些必要的细节被忽略了，请根据相关内容添加这些细节，使回答更加合理。
如果发现回答正确且不需要添加更多细节，直接输出原始回答即可。
** 重要提示 **
尽量保持修订回答中的结构
直接输出修订后的分析。除非被要求，否则不要在修订分析中添加额外的解释或声明。
################
你需要作答的题目如下: {}，
#################
刚才你已经完成了一部分分析，并且经过了我的检验，之前你完成内容如下：{}，
#################
当前步，你进行了下面的分析：{}，
#################
但是这一步的分析可能存在错误，因此我查询了一些相关资料
资料内容如下：{},
"""
REFINE_TEMPLATE_KEY =  ["question","past_thought","thought","reference"]

SUMMARY_PROMPT_TEMPLATE = """ 
你需要作答的题目如下: {}，
#################
刚才你已经完成了全部的【分析】，并且经过了我的检验，内容如下：{}，
请你根据之前的分析结果来推出最终结论
并写在【答案】和<eoa>之间。
完整的题目回答的格式如下：\n【分析】 ... <eoe>\n【答案】 ... <eoa>\n
请你严格按照上述格式作答。
你要作答的题目如下：{}
"""
SUMMARY_TEMPLATE_KEY = ["question","refine_thoughts","question"]


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
        "plain_prompt":["请你做一道地理选择题\n请你直接判断答案，不要写出思考过程，并将答案写在【答案】和<eoa>之间。完整的题目回答的格式如下：\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：{}",["question"]],
        "cot_prompt": ["请你做一道地理选择题，其中包含两到三个小题。\n请你一步一步思考。每一题你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：（1）【答案】 A <eoa>\n（2）【答案】 B <eoa>\n请你严格按照上述格式作答。\n{}",["question"]],
        "summary_prompt":["请你做一道地理选择题，题目如下: {}，你对ABCD一步一步的分析如下: A:{} B:{} C:{} D:{}\n 请根据之前几步的思考，从A，B，C，D中选出唯一正确的答案，并写在【答案】和<eoa>之间。\n回答的格式如下：：【答案】: ... <eoa>\n，再次重复题目：{}",["question","A","B","C","D","question"]]
    },
    "2010-2022_Political_Science_MCQs-single_choice":{
        "cot_prompt": ["请你做一道政治选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：{}",["question"]],
    },
    "2010-2022_Physics_MCQs-multi_choice":{
        "system_prompt": "你是一个非常优秀的高中生，正在作答一张物理试卷",
        "plain_prompt":["请你做一道物理选择题\n请你直接判断答案，不要写出思考过程，并将答案写在【答案】和<eoa>之间。完整的题目回答的格式如下：\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：{}",["question"]],
        "cot_prompt": ["请你做一道物理选择题。\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出所有符合题意的答案，并写在【答案】和<eoa>之间。\n例如：【答案】 AB <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】... <eoa>\n请你严格按照上述格式作答。\n{}",["question"]],
        "summary_prompt":["请你做一道物理选择题，题目如下: {}，你对ABCD一步一步的分析如下: A:{} B:{} C:{} D:{}\n 请根据之前几步的思考，从A，B，C，D中选出唯一正确的答案，并写在【答案】和<eoa>之间。\n回答的格式如下：：【答案】: ... <eoa>\n，再次重复题目：{}",["question","A","B","C","D","question"]],

    },
    "2010-2022_Chemistry_MCQs-single_choice": {
        "system_prompt": "你是一个非常优秀的高中生，正在作答一张化学试卷，现在需要你从A，B，C，D中选出唯一正确的答案，注意，只有一个正确答案",
        "plain_prompt":["请你做一道化学选择题\n请你直接判断答案，不要写出思考过程，并将答案写在【答案】和<eoa>之间。完整的题目回答的格式如下：\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：{}",["question"]],
        "cot_prompt": ["请你做一道化学选择题\n请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出唯一正确的答案，并写在【答案】和<eoa>之间。\n例如：【答案】: A <eoa>\n完整的题目回答的格式如下：\n【解析】 ... <eoe>\n【答案】 ... <eoa>\n请你严格按照上述格式作答。\n题目如下：{}",["question"]],
        "summary_prompt":["请你做一道化学选择题，题目如下: {}，你对ABCD一步一步的分析如下: A:{} B:{} C:{} D:{}\n 请根据之前几步的思考，从A，B，C，D中选出唯一正确的答案，并写在【答案】和<eoa>之间。\n回答的格式如下：：【答案】: ... <eoa>\n，再次重复题目：{}",["question","A","B","C","D","question"]]
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
    "2010-2022_Physics_MCQs-multi_choice":"physics",
    "2010-2022_Chemistry_MCQs-single_choice":"chemistry",
    "2010-2022_Geography_MCQs-multi_question_choice":"geography",
}

def get_query(inputs:dict)->str:
    return inputs["question"]


def gaokao_obj_run(subset_name:str,method:str,model_name="DeepSeek-V3",log_fold:str = Path("logs"),dl:DatasetLoader = DatasetLoader(),test:bool=False,api_base:str="https://api.deepseek.com/v1"):
    dataset = dl.get_dataset("gaokao_obj")
    test_dataset = dataset[subset_name]["test"]
    store_fold_path=log_fold/"gaokao_obj"/method/model_name/f"{subset_name}.jsonl"
    store_fold_path.parent.mkdir(parents=True,exist_ok=True)
    command = {
        "temperature":0.7,
        "max_tokens":1024,
    }
    if model_name == "DeepSeek-V3":
        command.update({
            "api_key":os.environ["DEEPSEEK_API_KEY"],
            "api_base":"https://api.deepseek.com/v1",
            "model_id":"deepseek-chat",
        })
    else:
        command.update({
            "api_key":"EMPTY",
            "api_base":api_base,
            "model_id":model_name,
        })
    # prepare for run
    wrapper = None
    if method == "plain":
        wrapper = partial(Method.plain,command=command,
                          system_prompt=TEMPLATE[subset_name]["system_prompt"],
                          input_template=TEMPLATE[subset_name]["plain_prompt"][0],input_template_keys=TEMPLATE[subset_name]["plain_prompt"][1])
    elif method == "cot":
        wrapper = partial(Method.plain,command=command,
                          system_prompt=TEMPLATE[subset_name]["system_prompt"],
                          input_template=TEMPLATE[subset_name]["cot_prompt"][0],input_template_keys=TEMPLATE[subset_name]["cot_prompt"][1])
    elif method == "rag":
        
        wrapper = partial(Method.rag,command=command,
                                     system_prompt=TEMPLATE[subset_name]["system_prompt"],
                                     input_template=TEMPLATE[subset_name]["cot_prompt"][0],input_template_keys=TEMPLATE[subset_name]["cot_prompt"][1],
                                     get_query_fn=get_query,retriever_name=RAG_SYSTEM_MAP[subset_name])
    elif method == "rat":
        wrapper = partial(Method.rat,command=command,
                                     system_prompt=TEMPLATE[subset_name]["system_prompt"],
                                     split_fn=partial(llm_split,command=command,verbos=test),
                                     get_query_fn=get_query_naive,
                                     retrieve_fn=partial(rag_retrieve,retriever_name=RAG_SYSTEM_MAP[subset_name],verbos=test),
                                     draft_prompt_template=TEMPLATE[subset_name]["cot_prompt"][0],draft_template_keys=TEMPLATE[subset_name]["cot_prompt"][1],
                                     refine_prompt_template=REFINE_PROMPT_TEMPLATE,refine_template_keys=REFINE_TEMPLATE_KEY,
                                     summary_prompt_template=SUMMARY_PROMPT_TEMPLATE,summary_template_keys=SUMMARY_TEMPLATE_KEY,
                                     verbos=test,
                                     )
    else:
        raise ValueError(f"unknown method: {method}")
    
    # run
    if test:
        # 将输出打印到命令行中
        import sys
        with open(store_fold_path.parent/'log.txt', 'w') as f:
            original_stdout = sys.stdout
            sys.stdout = f
            output = wrapper(test_dataset[112])
            sys.stdout = original_stdout
        return
    mp_utils.get_multiple_response(wrapper,[test_dataset[idx] for idx in range(test_dataset.num_rows)],batch_size=40,store_fold_path=store_fold_path,slow=True)
    return 
        
def gaokao_obj_test(subset_name:str,method:str,model_name="DeepSeek-V3",log_fold:PosixPath=Path("logs"),dl:DatasetLoader = DatasetLoader()):
    dataset = dl.get_dataset("gaokao_obj")
    test_dataset = dataset[subset_name]["test"]
    data_path = log_fold/"gaokao_obj"/method/model_name/f"{subset_name}.jsonl"
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',type=str,default="Llama3-8B-Chinese-Chat")
    parser.add_argument('--api_base',type=str,default="http://10.0.2.26:9401/v1")
    args = parser.parse_args()
    api_base = args.api_base
    model_name = args.model_name   #"Llama3-8B-Chinese-Chat"
    
    dl = DatasetLoader()
    
    sub_tasks = [
         "2010-2022_Chemistry_MCQs-single_choice",
         "2010-2022_Physics_MCQs-multi_choice",
         "2010-2022_Geography_MCQs-multi_question_choice",
    ]
    methods = [
        "plain",
        "cot",
        "rag",
        "rat",
    ]
    for method in methods:
        for sub_task in sub_tasks:
            gaokao_obj_run(sub_task,method=method,model_name=model_name,dl=dl,api_base=api_base)#test=True)
            gaokao_obj_test(sub_task,method=method,model_name=model_name,dl=dl)
