import torch
from openxlab.model import download
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

class Model_center():
    """
    存储检索问答链的对象 
    """
    def __init__(self, base_model_dir = '/home/model/internlm', adapter_dir = '/home/model/adapter', megered_dir = '/home/model/merged'):
        
        self.base_model_dir = base_model_dir
        self.adapter_dir = adapter_dir
        self.megered_dir = megered_dir
        # 加载模型
        self.load_model()

    def load_model(self):
        # 下载InternLM-chat-7B, self adapter
        if not os.path.exists(self.base_model_dir):
            download(model_repo='OpenLMLab/InternLM-chat-7b', output=self.base_model_dir)
        if not os.path.exists(self.adapter_dir):
            download(model_repo='xianyiliu/goog', output=self.adapter_dir)

        if not os.path.exists(self.megered_dir):
            os.system(f'xtuner convert merge {self.base_model_dir} {self.adapter_dir} {self.megered_dir} --max-shard-size 2GB ')
        
        # 加载InternLM-chat-7B
        print("loading...")
        self.model = AutoModelForCausalLM.from_pretrained(self.megered_dir, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map='auto')

        # 加载Tokenization
        self.tokenizer = AutoTokenizer.from_pretrained(self.megered_dir, trust_remote_code=True)



    def qa_self_answer(self, question: str, chat_history: list = []):
        """
        调用问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            response, history = self.model.chat(self.tokenizer, question, chat_history)
            chat_history.append((question, response))
            # 将问答结果直接附加到问答历史中，Gradio 会将其展示出来
            return "", chat_history
        except Exception as e:
            return e, chat_history

import gradio as gr

# 实例化核心功能对象
model_center = Model_center()
# 创建一个 Web 界面
block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):   
        with gr.Column(scale=15):
            # 展示的页面标题
            gr.Markdown("""<h1><center>InternLM</center></h1>
                <center>书生浦语</center>
                """)

    with gr.Row():
        with gr.Column(scale=4):
            # 创建一个聊天机器人对象
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                # 创建提交按钮。
                db_wo_his_btn = gr.Button("Chat")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")
                
        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(model_center.qa_self_answer, inputs=[
                            msg, chatbot], outputs=[msg, chatbot])

    gr.Markdown("""提醒：<br>
    1. 初始化数据库时间可能较长，请耐心等待。
    2. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)
gr.close_all()
# 直接启动
demo.launch()