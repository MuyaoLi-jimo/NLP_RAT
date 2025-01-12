from pathlib import Path,PosixPath
from src.utils import file_utils
import pandas as pd


def compare_multi_methods(task_name:str,model_name:str,sub_task:str,log_fold:PosixPath=Path("logs"),):
    log_path = log_fold / task_name 
    methods_name = [child.name for child in log_path.iterdir() if child.is_dir()]   
    methods_paths = {child.name: child / model_name / f"{sub_task}.jsonl" for child in log_path.iterdir() if child.is_dir()}
    score_board = {method_name: {} for method_name in methods_name}
    for method_name,method_path in methods_paths.items():
        method_file = file_utils.load_jsonl(method_path)
        for answer in method_file:
            answer_id = answer.get("id")
            answer_success = int(answer.get("success",0))
            score_board[method_name][answer_id[-3:]] = answer_success
    
    df = pd.DataFrame.from_dict(score_board, orient='index')
    df['total'] = df.mean(axis=1) 
    print(model_name)
    print(df)
    return df
    
def show_task(task_name:str,sub_task:str,log_fold:PosixPath=Path("logs"),):
    model_names = [
        "DeepSeek-V3",
        "Llama3-8B-Chinese-Chat",
        "qwen2-vl-72b-instruct",
    ]
    for model_name in model_names:
        df = compare_multi_methods(task_name,model_name,sub_task,log_fold)
    return
    
if __name__ == "__main__":
    # 2010-2022_Geography_MCQs-multi_question_choice
    # 2010-2022_Chemistry_MCQs-single_choice
    # 2010-2022_Physics_MCQs-multi_choice
    
    show_task("gaokao_obj","2010-2022_Chemistry_MCQs-single_choice")