from src.utils import api_utils
from src.Method.rag import RAG_SYSTEM



SYSTEM_PROMPT = "你是一个专业的问答机器人，旨在为用户提供准确和有用的信息。请根据用户的问题，给出直接、简洁且相关的回答。\n"

SPLIT_PROMPT = """ 
请帮我重新格式化以下思维链，将其每一步思考过程使用两个换行符'\n\n'分开。
请不要修改思维链的内容或添加任何额外的文本。只需在适当的地方插入换行符，以确保每个思考步骤都被清楚地隔开。
这包括在思考链的开头和每个选项之间插入换行符，以及在结尾之前。
例子：
step1\n\nstep2\n\nstep3 
** 重要提示 **
直接输出修订后的回答。除非被要求，否则不要在修订回答中添加额外的解释或声明。
下面是需要处理的原始思维链:
"""


QUERY_PROMPT = """
我想验证这段回答内容的正确性，需要你确定需要查询的内容。
请按照如下格式给出查询内容：
【查询】: ...
** 重要提示 **
请你严格按照上述格式作答。
直接输出查询。不要添加额外的解释或介绍。
带查询的回答如下：
"""

###################################
# split方法

def llm_split(draft_answer:str,command:dict,verbos:bool=False):
    """使用llm来分割原始问题"""
    conversations = [
        api_utils.create_system_message(SYSTEM_PROMPT),
        api_utils.create_user_message(SPLIT_PROMPT+draft_answer),
    ]
        # 包装好给generate_qa的输入
    data = {
        "messages":conversations,
    }    
    data.update(command)
    # 获取答案
    content,token = api_utils.generate_qa(data,verbos=verbos)
    split_contents = direct_split(content).get("split_contents")
    output = {
        "split_contents":split_contents,
        "token":token
    }
    return output

def choice_split(draft_answer:str, split_char:str = '\n\n',verbos:bool=False):
    """将draft根据split_char切分为多个段落
    只保留涉及到选项的那些段落
    """
    draft_paragraphs = draft_answer.split(split_char)
    filtered_draft_paragraphs = [paragraph for paragraph in draft_paragraphs if re.search(r'[A-Z]', paragraph)]
    output = {
        "split_contents":filtered_draft_paragraphs,
    }
    return output

def direct_split(draft_answer:str, split_char:str = '\n\n',verbos:bool=False):
    """将draft切分为多个段落"""
    draft_paragraphs = draft_answer.split(split_char)
    # print(f"The draft answer has {len(draft_paragraphs)}")
    output = {
        "split_contents":draft_paragraphs,
    }
    return output

##############################

def get_query_naive(thought:str,verbos:bool=False):
    """直接用每一步的thought来作为query，并用规则排除开头结尾
    """
    if "【解析】" in thought and len(thought) < 10:
        return None
    elif "【答案】" in thought and len(thought) < 10:
        return None
    elif "<eoe>" in thought and len(thought) < 10:
        return None 
    elif "<eoa>" in thought and len(thought) < 10:
        return None 
    elif "综上所述" in thought:
        return None
    elif "逐一分析" in thought or "逐步分析" in thought:
        return None
    elif "分析每个选项" in thought:
        return None
    output = {
        "query":thought
    }
    return output

#############################

def rag_retrieve(query:str,retriever_name:str,verbos:bool=False):
    """用rag来进行检索
    """
    rag_system = RAG_SYSTEM()
    rag_system.get_retriever(retriever_name) 
    docs = rag_system.retrieve(query)
    if verbos:
        print(docs[0].page_content)
        print("!!!!!!!!!!!!!!!!!!!!")
    output = {
        "ref_content":docs[0].page_content,
    }
    return output
