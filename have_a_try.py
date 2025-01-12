import gradio as gr
from examples.techniques import rat
from openai import OpenAI
import random
from datetime import datetime
import os
# 爬网页用的
os.environ["USER_AGENT"] = 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.99 Safari/537.36'

random.seed(2024)


def rat_n(api_key, question):
    # 系统prompt
    system_prompt = f"你是deepseek，一个专业的问答机器人，旨在为用户提供准确和有用的信息。请根据用户的问题，给出直接、简洁且相关的回答。今天是: {datetime.now().strftime('%Y-%m-%d')}。\n"
    # 准备api
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )
    return rat(client, system_prompt, question)[1]


def chatbot_interface(question, api_key):
    response = rat_n(api_key, question)
    return response

# 界面
iface = gr.Interface(
    fn=chatbot_interface,  
    inputs=["text", "text"],  # 输入：问题和 API Key
    outputs="text",
    live=False,
    title="RAT问答机器人",
    description="输入你的问题和 DeepSeek API Key，获取回答。",  # 界面描述
)


iface.launch()
