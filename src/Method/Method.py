"""
API for different methods
"""

from src.utils import api_utils
from src.Method.rag import RAG_SYSTEM
from typing import Callable
import random



def plain(inputs:dict,system_prompt,command:dict,
          input_template:str,input_template_keys:list,):
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
    content,token = api_utils.generate_qa(data)
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

def rat(inputs:dict,system_prompt,command:dict,
        input_template:str,input_template_keys:list,
        get_query_fn:Callable,retriever_name:str,
        ):
    pass

if __name__ == "__main__":
    pass