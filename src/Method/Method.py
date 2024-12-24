"""
API for different methods
"""

from src.utils import api_utils
from src.Method.rag import RAG_SYSTEM
from typing import Callable
import random
from copy import deepcopy


def plain(inputs:dict,system_prompt,command:dict,
          input_template:str,input_template_keys:list,
          verbos=False):
    """use plain prompt
    输入dataset，prompt template
    输出final answer
    :command 给大模型的指令，包括温度，topk，...，
    """
    # 使用input字典中相应的值来填充模板
    formatted_values = [inputs[key] for key in input_template_keys]
    user_prompt = input_template.format(*formatted_values)
    conversations = [
        api_utils.create_system_message(system_prompt),
        api_utils.create_user_message(user_prompt)
    ]
    # 包装好给generate_qa的输入
    data = {
        "messages":conversations,
    }
    data.update(command)
    # 获取答案
    content,token = api_utils.generate_qa(data,verbos=verbos)
    output = {
        "id":inputs["id"],
        "answer":content,
        "token":token
    }
    return [output]

def cot(inputs:dict,system_prompt,command:dict,
        input_template:str,input_template_keys:list,
        ):
    """use Chain-of-Thought"""
    formatted_values = [inputs[key] for key in input_template_keys]
    user_prompt = input_template.format(*formatted_values)
    user_prompt += "\nLet's think step by step:"
    conversations = [
        api_utils.create_system_message(system_prompt),
        api_utils.create_user_message(user_prompt)
    ]
    # 包装好
    data = {
        "messages":conversations,
    }
    data.update(command)
    content,token = api_utils.generate_qa(data)

    output = {
        "id":inputs["id"],
        "answer":content,
        "token":token
    }
    
    return [output]
    
def icl(inputs:dict,system_prompt,k:int,command:dict,
        input_template:str,input_template_keys:list,
        output_template:str,output_template_keys:list,
        examples:list,):
    """use few shot learning"""
    assert k<=5

    user_prompt = "#########################\n"
    # 增加随机性
    select_examples = random.sample(examples,k)
    # 把例子写在前面
    for example in select_examples:
        question_format_values = [example[key] for key in input_template_keys]
        answer_format_values = [example[key] for key in output_template_keys]
        question = input_template.format(*question_format_values)
        answer =  output_template.format(*answer_format_values)
        user_prompt += question + answer

    user_prompt += """##########################
    According to the examples above, answer the following question: 
    """
    formatted_values = [inputs[key] for key in input_template_keys]
    user_prompt += input_template.format(*formatted_values)
    
    conversations = [
        api_utils.create_system_message(system_prompt),
        api_utils.create_user_message(user_prompt),
    ]
    data = { "messages":conversations, }
    data.update(command)
    content,token = api_utils.generate_qa(data)
    if content.lstrip().startswith("Answer:"):
        start_pos = content.lstrip().find("Answer:") + len("Answer:")
        content = content[start_pos:].lstrip()
    output = {
        "id":inputs["id"],
        "answer":content,
        "token":token
    }
    
    return [output]
    
def reflexion(inputs:dict,system_prompt,command:dict,
    input_template:str,input_template_keys:list,
    reflex_input_template:str,
    interact_fc:Callable[[str,dict],str],roll:int=1,
):
    """use reflexion 
    :datahelper which help to example
    :roll the max reflexion times
    """
    conversations = [
        api_utils.create_system_message(system_prompt),
    ]
    formatted_values = [inputs[key] for key in input_template_keys]
    user_prompt = input_template.format(*formatted_values)
    conversations.append(api_utils.create_user_message(user_prompt))
    data = { "messages":conversations, }
    data.update(command)
    content,token = api_utils.generate_qa(data)
    success_flag, info= interact_fc(content,inputs)
    if success_flag:
        output = {
            "id":inputs["id"],
            "answer":content,
            "token":token
        }
        return [output]
    
    roll_idx = roll
    while roll_idx:
        reflex_prompt = reflex_input_template.format(info)
        conversations.extend([api_utils.create_assistant_message(content),
                             api_utils.create_user_message(reflex_prompt)])
        data = { "messages":conversations, }
        data.update(command)
        content,new_token = api_utils.generate_qa(data)
        token = api_utils.acc_tokens(token,new_token)
        success_flag, info= interact_fc(content,inputs)
        if success_flag:
            break
        roll_idx -= 1
        
    output = {
        "id":inputs["id"],
        "answer":content,
        "token":token
    }
    return [output]
    
def rag(inputs:dict,system_prompt,command:dict,
        input_template:str,input_template_keys:list,
        get_query_fn:Callable,retriever_name:str,
        ):
    """使用封装好的rag_system首先对query进行检索 """
    formatted_values = [inputs[key] for key in input_template_keys]
    user_prompt = input_template.format(*formatted_values)
    rag_system = RAG_SYSTEM()
    rag_system.get_retriever(retriever_name) 
    docs = rag_system.retrieve(get_query_fn(inputs))
    rag_prompt="这些是可以参考的资料: \n"
    
    for idx,doc in enumerate(docs):
        rag_prompt += "##################\n 参考资料" + str(idx) + ":\n"
        rag_prompt += doc.page_content + "\n"
    rag_prompt += "##################\n"
    user_prompt = rag_prompt + user_prompt
    conversations = [
        api_utils.create_system_message(system_prompt),
        api_utils.create_user_message(user_prompt)
    ]
    # 包装好给generate_qa的输入
    data = {
        "messages":conversations,
    }
    data.update(command)
    # 获取答案
    content,token = api_utils.generate_qa(data)
    output = {
        "id":inputs["id"],
        "answer":content,
        "token":token
    }
    return [output]

def rat(inputs:dict,command:dict,
        system_prompt:str,
        split_fn:Callable,
        get_query_fn:Callable,
        retrieve_fn:Callable,
        draft_prompt_template:str,draft_template_keys:list,
        refine_prompt_template:str,refine_template_keys:list,
        summary_prompt_template:str,summary_template_keys:list,
        verbos=False
        ):
    """use rat method to get the refined chain-of-thought and the final answer
    Args:
        inputs (dict): 输入的材料
        command (dict): LLM的命令设置
        system_prompt (str): 系统提示信息
        split_fn (Callable): 用于分割初稿的函数
        get_query_fn (Callable): 用于生成查询的函数
        retrieve_fn (Callable): 用于检索信息的函数
        draft_prompt_template (str): 草稿提示模板
        draft_template_keys (list): 草稿模板键
        refine_prompt_template (str): 修改单步思考模板
        refine_template_keys (list): 修改单步思考模板键
        summary_prompt_template (str): 总结提示模板
        summary_template_keys (list): 总结模板键
        
    """
    history = {} #记录整个过程
    total_tokens = {"input":0,"output":0,}
    # 首先用LLM完成初稿
    draft_outputs = plain(inputs,system_prompt,command,draft_prompt_template,draft_template_keys,verbos=verbos)
    draft_answer = draft_outputs[0]["answer"]
    history["draft"] = draft_answer
    count_token(total_tokens,draft_outputs[0]["token"])
    
    # 然后用提供的方法对原始输入进行分割
    split_thoughs_outputs = split_fn(draft_answer)
    thoughts = split_thoughs_outputs["split_contents"]
    history["split_contents"] = thoughts
    count_token(total_tokens,split_thoughs_outputs.get("token",{}))
    history["steps"] = []
    
    # 对每个thought进行检索-修改
    refine_thoughts = ""
    for idx,thought in enumerate(thoughts):
        # 制作query
        query_output = get_query_fn(thought)
        if query_output is None:
            continue
        query = query_output["query"]
        count_token(total_tokens,query_output.get("token",{}))
        
        # 检索
        ref_content_output = retrieve_fn(query)
        ref_content = ref_content_output["ref_content"]
        count_token(total_tokens,ref_content_output.get("token",{}))
        
        # 根据检索信息修改答案
        refine_inputs = {"question":inputs["question"],"past_thought":refine_thoughts,"thought":thought,"reference":ref_content,"id":inputs["id"]}
        refine_thought_outputs = plain(refine_inputs,system_prompt,command,refine_prompt_template,refine_template_keys,verbos=verbos)
        refine_thought = refine_thought_outputs[0]["answer"]
        count_token(total_tokens,refine_thought_outputs[0].get("token",{}))
        refine_thoughts +=  refine_thought + "\n\n"
        
        history["steps"].append({
            "query":query,
            "ref_content":ref_content,
            "origin_thought":thought,
            "refine_thought":refine_thought,
        })
    
    summary_inputs = {"question":inputs["question"],"refine_thoughts":refine_thoughts,"id":inputs["id"]}
    final_answer_outputs = plain(summary_inputs,system_prompt,command,summary_prompt_template,summary_template_keys,verbos=verbos)
    final_answer = final_answer_outputs[0]["answer"]
    history["final_answer"] = final_answer
    count_token(total_tokens,final_answer_outputs[0]["token"])
    
    output = {
        "id":inputs["id"],
        "answer":final_answer,
        "token":total_tokens,
        "history":history,
    }
    return [output]
    
def count_token(total_tokens,new_tokens):
    total_tokens["input"] += new_tokens.get("input",0)
    total_tokens["output"] += new_tokens.get("output",0)
    return total_tokens


if __name__ == "__main__":
    pass