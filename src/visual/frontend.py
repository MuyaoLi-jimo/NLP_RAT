import gradio as gr
from src.visual.visual import compare_multi_methods

SUBTASK_MAP = {
    "chemistry":"2010-2022_Chemistry_MCQs-single_choice"
}

def compare_multi_methods_wrapper(task_subtask:str):
    task_subtask = str(task_subtask.value)
    task_name = task_subtask.split("-")[0]
    print(task_subtask.split("-"))
    sub_task = SUBTASK_MAP.get(task_subtask.split("-")[1])
    return  compare_multi_methods(task_name,sub_task)

with gr.Blocks(title="RAT leaderboard") as page:
    setting_drop = gr.Dropdown(
        ["gaokao_obj-chemistry"], 
        label="benchmark", 
        value = "gaokao_obj-chemistry",
        info="Which benchmark do you choose?",
        interactive=True
    )
    leaderboard = gr.DataFrame(compare_multi_methods("gaokao_obj",SUBTASK_MAP["chemistry"]))
    setting_drop.change(compare_multi_methods_wrapper,
                            inputs=[setting_drop],
                            outputs=[leaderboard])



page.launch(share=True,
            #auth=("admin", "craftjarvis"),
            server_port=8081,)