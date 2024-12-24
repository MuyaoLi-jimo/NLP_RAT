import os
import requests
import json
from langchain_community.document_transformers import Html2TextTransformer
from langchain_community.document_loaders import AsyncHtmlLoader
import re
from openai import OpenAI


MODEL = "deepseek-chat"

# 初始CoT
draft_prompt = """
** 重要提示 **
尝试一步一步地思考并回答这个问题/指示，使答案更具结构性。
使用 '\n\n' 将答案分成几个段落。
直接回应指示。除非被要求，否则不要在答案中添加额外的解释或介绍。
"""

# 反思Prompt
revise_prompt = """
我希望根据检索到的与问题相关的网页内容来修订答案。
你需要检查答案是否正确。
根据网页内容，如果发现答案中有错误，请修改答案。
如果发现有些必要的细节被忽略了，请根据相关内容添加这些细节，使答案更加合理。
如果发现答案正确且不需要添加更多细节，直接输出原始答案即可。
** 重要提示 **
尽量保持修订答案中的结构（即带有副标题的多个段落，请不要改变原来"答案"的段落数目）并使其更具结构性，便于理解。
使用 '\n\n' 字符分隔段落。
直接输出修订后的答案。除非被要求，否则不要在修订答案中添加额外的解释或声明。
"""

# 查询Prompt
query_prompt = """
我想验证给定问题的答案的内容正确性，特别是最后一段话，需要你确定上网查询的内容。
- 如果你需要查询某个城市的天气情况，请输出：
{
    "type":"天气",
    "cities": [列表内是你要查询天气的城市]
}
- 如果你需要查询百度百科，请输出：
{
    "type":"百科",
    "items":[列表内是你想查询的所有百科词条名称]
}
- 如果你认为无需查询，请输出：
{
    "type":"无"
}
尽量使查询与答案内容最后一段尽可能相关。
** 重要提示 **
直接输出查询(必须遵照JSON格式)。不要添加额外的解释或介绍。
"""

# 寻找参考文段Prompt
reference_prompt = """
我想验证给定问题的答案的内容正确性，特别是最后几句话，在百科上查询到了若干词条，目录如上。
请根据问题和答案，输出你想要的参考内容路径，路径深度不限，格式如下：
词条名->一级标题->二级标题
词条名->一级标题
词条名->一级标题->二级标题->三级标题
...
尽量使参考内容路径与答案最后几句话尽可能相关。
** 重要提示 **
直接按格式输出参考内容路径，每行一条。不要添加额外的解释或介绍。
"""





def get_weather(city:str):
    """
    Args:
        city (str): 查询的城市
    Returns:
        str: 天气字符串
    """
    
    weather_url = "https://query.asilu.com/weather/gaode"
    params = {
        "city":city
    }
    response = requests.get(url=weather_url, params=params)
    weather_str = ""
    if response.json()["infocode"]=="10000":
        for cast in response.json()["forecasts"][0]["casts"]:
            weather_str += f"""日期：{cast["date"]}：日间天气：{cast["dayweather"]}，日间气温：{cast["daytemp"]}，日间风向为{cast["daywind"]}，风力{cast["daypower"]}；夜间天气：{cast["nightweather"]}，夜间气温：{cast["nighttemp"]}，夜间风向为{cast["nightwind"]}，风力{cast["nightpower"]}。\n"""
    else:
        weather_str = "没有检索到天气信息。"
    return (f"根据天气预报，这是{city}今天和后三天的天气：\n" + weather_str)


def get_baike(name:str):
    """
    Args:
        name (str): 百科词条名称
    Returns:
        json: 百科json树
        搜索不到返回None
    """
    def check_file_exists(directory, filename):
        # 检查文件是否存在
        for root, dirs, files in os.walk(directory):
            if filename in files:
                return os.path.join(root, filename)
        return None
    # 指定目录和文件名
    directory_path = './baike/'
    file_name = f'{name}.json'
    # 检查是否存在已有的json
    file_path = check_file_exists(directory_path, file_name)
    
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            json_output = json.load(f)
    else:
        # 没有，上网搜索
        link = 'https://baike.baidu.com/item/' + name
        page_content = get_page_content(link)
        if page_content != None:
            json_output = markdown_to_json(page_content)
            if json_output['title'] == "百度百科错误页":
                json_output = None
            else:
                #print(json.dumps(json_output, ensure_ascii=False))
                with open(directory_path+file_name, "w", encoding="utf-8") as f:
                    json.dump(json_output, f, ensure_ascii=False, indent=4)
        else:
            json_output = None
            
    return json_output
        



def get_page_content(link:str):
    """
    Args:
        link (str): 网址
    Returns:
        str: 网页内容字符串,组织为Markdown格式(用# ## 分割标题)
        None: 如果网页不存在
    """
    print(f">>>检索中：{link}")
    # 需要去除的字符串
    substring_1 = "\n新手上路\n\n成长任务编辑"
    substring_2 = "\n播报讨论上传视频"
    
    loader = AsyncHtmlLoader([link])
    docs = loader.load()
    html2text = Html2TextTransformer()
    docs_transformed = html2text.transform_documents(docs)
    if len(docs_transformed) > 0:
        pos = docs_transformed[0].page_content.find(substring_1)
        clean_string = docs_transformed[0].page_content[:pos].replace(substring_2, '')
        print("<<<检索成功")
        return clean_string
    else:
        print("<<<检索失败")
        return None




def markdown_to_json(markdown_text):
    """
    Args:
        markdown_text (_type_): 含有markdown小标题的文本
    Returns:
        json: 按标题层次转换为json(树状结构)
    """
    # 使用正则表达式匹配Markdown小标题
    headers_pattern = re.compile(r'^(#+)\s+(.*)')
    content_lines = markdown_text.split('\n')
    
    # 初始化JSON结构
    json_output = {'title': 'root', 'content': '', 'children': []}
    current_level = [json_output, 0, []]  # 当前层级、标题级别和内容列表
    stack = []

    for line in content_lines:
        header_match = headers_pattern.match(line)
        if header_match:
            # 获取小标题的级别和内容
            level = len(header_match.group(1))
            title = header_match.group(2)
            
            # 根据级别调整current_level的位置
            while stack and stack[-1][1] >= level:
                current_level = stack.pop()
            
            # 创建新的字典并设置为current_level
            new_level = {'title': title, 'content': '', 'children': []}
            
            # 如果当前层级是更深的层级，则更新current_level
            if level > current_level[1]:
                current_level[0]['children'].append(new_level)
                stack.append(current_level)
                current_level = [new_level, level, []]
            else:
                # 如果是同一层级或更浅的层级，则添加到父级的children中
                if level == current_level[1]:
                    stack[-1][0]['children'].append(new_level)
                else:
                    current_level[0]['children'].append(new_level)
                stack.append(current_level)
                current_level = [new_level, level, []]
        else:
            # 处理内容行，将内容添加到当前级别的键值对中
            if current_level[0]:
                current_level[0]['content'] += (line + '\n')

    # 将最终的栈中的内容加入到json_output中
    while stack:
        current_level = stack.pop()
        for child in current_level[2]:
            current_level[0]['children'].append(child)

    return json_output['children'][0]  # 返回第一个元素，因为我们的结构是从根开始的



def get_draft(client:OpenAI, system_prompt:str, question):
    """
    Args:
        client (OpenAI): 大模型
        system_prompt (str)
        question (str): 问题
    Returns:
        str: 初始cot答案
    """
    draft = client.chat.completions.create(
        model=MODEL,
        messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"##问题: {question}\n\n##指示: {draft_prompt}"
                }
            ],
            temperature = 1.0,
    ).choices[0].message.content
    return draft



def split_draft(draft, split_char = '\n\n'):
    """将draft切分为多个段落"""
    draft_paragraphs = draft.split(split_char)
    # print(f"The draft answer has {len(draft_paragraphs)}")
    return draft_paragraphs



def get_query(client:OpenAI, system_prompt:str, question, answer):
    """
    Args:
        client (OpenAI): 大模型
        system_prompt (str)
        question (str): 问题
        answer (str): 原始答案
    Returns:
        json: query是json格式的
    """
    query = client.chat.completions.create(
        model=MODEL,
        messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"##问题: {question}\n\n##答案: {answer}\n\n##指示: {query_prompt}"
                }
            ],
            temperature = 1.0,
            response_format={
                'type': 'json_object'
            }
    ).choices[0].message.content
    try:
        # 尝试将字符串转换为JSON
        query = json.loads(query)
        return query
    except json.JSONDecodeError:
        # 如果转换失败，返回错误信息
        return {"type": "error"}



def content_tree(node, depth=0):
    tree_str = ""
    # 打印当前节点的标题和内容，depth表示当前的层级深度
    indent = "    " * depth + "#" * depth  # 每层增加四个空格的缩进
    tree_str += f"{indent} {node['title']}\n"
    # 递归打印子节点
    for child in node['children']:
        tree_str += content_tree(child, depth + 1)
    # 返回树    
    return tree_str



def get_references(client:OpenAI, system_prompt:str, question, answer, ref_trees:list)->str:
    """_summary_
    Args:
        client (OpenAI): 大模型
        system_prompt (str)
        question (str): 问题
        answer (str): 原始答案
        ref_trees (list): 参考词条树结构的列表

    Returns:
        str: _description_
    """
    
    # ref_trees 变成 str
    ref_trees_str = "\n".join(ref_trees)
    ref_routes = client.chat.completions.create(
        model=MODEL,
        messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"##问题: \n{question}\n\n##答案: \n{answer}\n\n##词条目录: \n{ref_trees_str}\n\n##指令：\n{reference_prompt}"
                }
            ],
            temperature = 1.0
    ).choices[0].message.content
    
    return ref_routes



def routes_to_content(ref_routes:str, baike_jsons:list)->str:
    """_summary_

    Args:
        ref_routes (str): LLM生成的参考路径
        baike_jsons (list): 百科json的列表，每个元素是dict

    Returns:
        str: 参考内容
    """
    lines = ref_routes.splitlines()
    
    def get_baike_in_list(name, bklist):
        for bk in bklist:
            if bk['title'] == name:
                return bk
        return None
    
    contents = []        
    for line in lines:
        route_list = line.split('->')
        flag = True #False标记寻找失败
        bklist = baike_jsons #初始list是最外层
        bk = None
        for name in route_list:
            bk = get_baike_in_list(name, bklist)
            if bk == None:
                break
            else:
                bklist = bk['children']
        if bk != None:
            contents.append(json.dumps(bk, indent=4, ensure_ascii=False))
            
    return "\n".join(contents)
                
            
        
    

def get_revise_answer(client:OpenAI, system_prompt:str, question, answer, web_content):
    """
    Args:
        client (OpenAI): 大模型
        system_prompt (str)
        question (str): 问题
        answer (str): 原始答案
        web_content (str): 网页内容
    Returns:
        str: 反思后的答案
    """
    revised_answer = client.chat.completions.create(
        model=MODEL,
        messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"##问题:\n {question}\n\n##答案: \n{answer}\n\n##网页内容: \n{web_content}\n\n##指示: \n{revise_prompt}"
                }
            ],
            temperature = 1.0
    ).choices[0].message.content
    return revised_answer

