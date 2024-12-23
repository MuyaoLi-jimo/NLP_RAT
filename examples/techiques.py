import json
from openai import OpenAI
from rat_tools import *
from datetime import datetime



def cot(client:OpenAI, system_prompt:str, input):
    '''Chain of Thoughts'''
    
    # prompt
    user_prompt = """
########## 重要提示 #########:
尝试一步一步地思考并回答这个问题/指示，使答案更具结构性。
使用 '\n\n' 将答案分成几个段落。
直接回应指示。除非被要求，否则不要在答案中添加额外的解释或介绍。
    """
    
    # 生成回答
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": input},
                {"role": "user", "content": user_prompt}]
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=4096,
        temperature=0.5
    )
    
    # 计算token消耗
    p_tokens = response.usage.prompt_tokens
    c_tokens = response.usage.completion_tokens
    
    result = response.choices[0].message.content
    
    return result, (p_tokens, c_tokens)



def rat(client:OpenAI, system_prompt:str, question:str):
    
    # draft
    print(f"{datetime.now()} [INFO] 获取草稿...")
    draft = get_draft(client, system_prompt, question)
    print(f"{datetime.now()} [INFO] 返回草稿")
    print(f"##################### DRAFT #######################")
    print(draft)
    print(f"#####################  END  #######################\n\n")
    draft_paragraphs = split_draft(draft)
    print(f"{datetime.now()} [INFO] 草稿被切分为{len(draft_paragraphs)}部分")
    
    # 修改draft每个paragraph
    answer = ""
    for i, p in enumerate(draft_paragraphs):
        print(f"\n\n@@@@@@@@@@@@@@@@@@@@ PART {i+1}/{len(draft_paragraphs)} @@@@@@@@@@@@@@@@@@@@")
        print(f"{datetime.now()} [INFO] 修改第{i+1}/{len(draft_paragraphs)}部分...")
        answer = answer + '\n\n' + p
        # Query
        print(f"{datetime.now()} [INFO] 生成对应Query...")
        query = get_query(client, system_prompt, question, answer)
        
        # 修改
        if query['type'] == "天气":
            print(f"{datetime.now()} [INFO] 获取天气：{query['cities']}...")
            weather_info = ''
            for city in query['cities']:
                weather_info += (get_weather(city) + '\n')
            print(f"{datetime.now()} [INFO] 根据网页内容修改对应答案...")
            revised_answer = get_revise_answer(client, system_prompt, question, answer, weather_info)
            print(f"{datetime.now()} [INFO] 答案修改完成")
            
        elif query['type'] == "百科":
            # 获取
            print(f"{datetime.now()} [INFO] 尝试获取百科：{query['items']}...")
            baike_jsons = []
            for item in query['items']:
                baike = get_baike(item)
                if baike != None:
                    baike_jsons.append(baike)
            # 筛选&修改
            ref_trees = [content_tree(baike_json) for baike_json in baike_jsons]
            if len(ref_trees) == 0:
                # 处理没有百科词条的情况，answer不变
                print(f"{datetime.now()} [INFO] 没有找到百科词条：{query['items']}")
                revised_answer = answer
            else:
                # 筛选百科内容
                print(f"{datetime.now()} [INFO] 找到百科词条：{[baike_json['title'] for baike_json in baike_jsons]}，筛选参考内容中...")
                ref_routes = get_references(client, system_prompt, question, answer, ref_trees)
                print(f"{datetime.now()} [INFO] 筛选到以下参考内容:\n{ref_routes}")
                ref_content = routes_to_content(ref_routes, baike_jsons)
                print(f"{datetime.now()} [INFO] 参考内容文本如下:\n{ref_content}\n")
                print(f"{datetime.now()} [INFO] 根据网页内容修改对应答案...")
                revised_answer = get_revise_answer(client, system_prompt, question, answer, ref_content)
                print(f"{datetime.now()} [INFO] 答案修改完成")
                
        else:
            # 不需要参考内容修改 / LLM生成Query出错了
            revised_answer = answer
                
        # 替换原来的答案
        answer = revised_answer
        print(f"\n\n##################### ANSWER #######################")
        print(answer)
        print(f"#####################  END  #######################\n\n")
        
    return draft, answer
                
            




