from rich import print,console
from tqdm import tqdm
from typing import Dict
import numpy as np
from DataHelper import DatasetLoader
from datasets import DatasetDict
import re

def evaluate(dataset_name:str,answers:dict,sub_dataset_name="",method:str="main",dataset=None):
    """Evaluate the performance using provided answers.
    :param dataset_name: Name of the dataset to evaluate.
    :param answers: A dictionary of answers where keys are the question IDs.
    :param method: Evaluation method (default is 'main').
    """
    if len(answers)==0:
        return {}
    
    if dataset is None:
        dl = DatasetLoader()
        dataset = dl.get_dataset(dataset_name)
    
    if dataset_name=="gsm8k":
        #使用accuracy
        return math_evaluate(answers=answers,dataset=dataset,method=method)
    elif dataset_name=="mbpp":
        # 要求使用temperature=1.3 top p = 0.95的nucleus采样
        return code_evaluate(answers=answers,dataset=dataset,method=method)
    elif dataset_name =="gaokao_obj":
        return gaokao_obj(answers=answers,dataset=dataset,method=method,sub_dataset_name=sub_dataset_name)
    else:
        raise ValueError(f"no such dataset: {dataset_name}")

####################
#     choice       #
####################
def gaokao_obj(answers:dict,dataset:DatasetDict,sub_dataset_name:str,method:str="main",):
    """对gaokao数据集的一个子集测评 """
    question_type = sub_dataset_name.split("-")[-1]
    sub_dataset = dataset[sub_dataset_name]["test"]
    evaluated_count = 0
    scores = {}
    for item in tqdm(sub_dataset):
        question_id = item['id']
        if question_id not in answers:
            continue
        evaluated_count += 1
        model_raw_answer = answers[question_id]
        
        model_answer = extract_choice_answer(model_output=model_raw_answer,question_type=question_type)
        standard_answer = item.get("answer")
        question_score = item["score"]
        
        # 计算score
        scores[question_id] = scoring_answer(model_answer,standard_answer,question_score,question_type)
    return scores

def extract_choice_answer(model_output, question_type, answer_lenth=None):
    """
    Extract choice answer from model output

    Format of model_output that is expected:
    'single_choice': choice answer should be the last Capital Letter of the model_output, e.g.: "...【答案】 A <eoa>"
    'multi_question_choice': "...【答案】A ... 【答案】C ..." or write the choice answers at the beginning of the model_output, e.g. "A C D E F...."
    'multi_choice': "...【答案】 ABD " or write the choice answers at the end of the model_output, e.g. "... ACD"
    'five_out_of_seven': choice answers should be the first five Capital Letters of the model_output, e.g. "A C D F B ...."
    """
    if question_type == 'single_choice':
        model_answer = []
        temp = re.findall(r'[A-D]', model_output[::-1])
        if len(temp) != 0:
            model_answer.append(temp[0])

    elif question_type == 'multi_question_choice':
        answer_lenth = 10
        model_answer = []
        temp = re.findall(r"【答案】\s*[:：]*\s*[A-Z]", model_output)
            
        if len(temp) == answer_lenth:
            for t in temp:
                model_answer.append(re.findall(r'[A-Z]', t)[0])
        else:
            temp = re.findall(r"[A-Z]", model_output)
            if len(temp) > 0:
                for k in range(min(len(temp), answer_lenth)):
                    model_answer.append(temp[k])

    elif question_type == 'multi_choice':
        model_answer = []
        answer = ''
        content = re.sub(r'\s+', '', model_output)
        answer_index = content.find('【答案】')
        if answer_index > 0:
            temp = content[answer_index:]
            if len(re.findall(r'[A-D]', temp)) > 0:
                for t in re.findall(r'[A-D]', temp):
                    answer += t
        else:
            temp = content[-10:]
            if len(re.findall(r'[A-D]', temp)) > 0:
                for t in re.findall(r'[A-D]', temp):
                    answer += t
        if len(answer) != 0:
            model_answer.append(answer)
    
    elif question_type == 'five_out_of_seven':
        model_answer = []
        temp = re.findall(r'[A-G]', model_output)
        if len(temp) > 0:
            for k in range(min(5, len(temp))):
                model_answer.append(temp[k])

    return model_answer

def scoring_answer(model_answer:list,standard_answer:list,question_score:int,question_type:str):
    set_standard_answer = set(standard_answer)
    set_model_answer = set(model_answer)

    success = 0
    if question_type == "multi_question_choice" or question_type == "multi_choice":
        if set_model_answer==set_standard_answer:
            success = 1
        elif set_model_answer in set_standard_answer:
            success = 0.5
        else:
            success = 0
    elif question_type=="five_out_of_seven" or question_type== "single_choice":
        success = set_standard_answer == set_model_answer
    else:
        raise ValueError(f"unknow set {question_type}")
    score = int(question_score * success)
    return dict(score=score,success=success)
        

#####################
#      math         #
#####################

def math_evaluate(answers:dict,dataset:DatasetDict,method:str="main",):
    correct_count = 0
    evaluated_count = 0
    for item in dataset["test"]:
        question_id = item['id']
        if question_id in answers:
            print(question_id,answers)
            user_answer = answers[question_id]
            correct_answer = item['gt']
            if user_answer == correct_answer:
                correct_count += 1
            evaluated_count += 1
    accuracy = correct_count / evaluated_count 
    console.Console().log(f"Accuracy: {accuracy:.2f}")
    return accuracy

#####################
#      code         #
#####################

def code_evaluate(answers:dict,dataset:DatasetDict,method:str="main",):
    """ 
    检验code的结果
    """
    response = {}
    if method=="main" or method == "pass@1" or method=="accuracy":
        # 用Pass@k来评价，这里我们限定k=1，n=1
        evaluated_count = 0
        pass_k_accumulate = 0
        for item in tqdm(dataset["test"]):
            question_id = item['id']
            if question_id in answers:
                evaluated_count += 1
                user_answer = answers[question_id]
                success=pass_k(1,mbpp_examiner(user_answer,item)[0],)
                response[question_id]=success
                pass_k_accumulate+=success
        pass_k_score = pass_k_accumulate/evaluated_count
        console.Console().log(f"Pass@k score: {pass_k_score:.2f}")
    return response

def pass_k(num_samples:int,num_correct:int,k:int=1):
    """ 
    Calculates 1 - comb(n - c, k) / comb(n, k).
    """
    if num_samples - num_correct < k:
        return 1.0
    return  1.0 - np.prod(1.0 - k / np.arange(num_samples - num_correct + 1, num_samples + 1))  

def mbpp_examiner(user_answer,inputs):
    text_list = inputs["test_list"]
    user_answer = program_filter(user_answer)
    check_program = user_answer + "\n" + inputs.get("test_setup_code","") + "\n".join(text_list)
    return program_test_wrapper(check_program)

def program_filter(completion: str) -> str:
    # The program tends to overwrite, we only take the first function
    
    code_start = completion.find("```python")
    code_end = completion.find("```", code_start + len("```python"))
    if code_start != -1 and code_end != -1:
        completion = completion[code_start + len("```python"):code_end].strip()
    
    completion = completion.lstrip("\n")
    completion.replace("\t", "    ")
    return completion.split("\n\n")[0]

import multiprocessing

def target_function(check_program, conn):
    # 调用实际的 program_test
    success, message = program_test(check_program)
    conn.send((success, message))  # 发送结果到父进程
    conn.close()

def program_test_wrapper(check_program: str):
    # 子进程


    parent_conn, child_conn = multiprocessing.Pipe()
    process = multiprocessing.Process(target=target_function, args=(check_program, child_conn))
    process.start()
    process.join(timeout=5)  # 设置最大运行时间为5秒

    if process.is_alive():
        process.terminate()  # 如果代码仍在运行，则终止进程
        process.join()
        return False, "Timeout, Execution took too long"

    if parent_conn.poll():  # 检查管道中是否有数据
        success, message = parent_conn.recv()
        return success, message
    else:
        return False, "No response from child process"

def program_test(check_program):
    try: 
        exec(check_program)
        return True, "Execution successful"
    except Exception as e:
        return False, str(e)
    