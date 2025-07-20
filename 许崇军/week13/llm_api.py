#利用OpenAI API 调用远端大模型API，调试参数观察输出结果的差异。
import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

if __name__ == '__main__':
     load_dotenv(find_dotenv())
     client = OpenAI(
         api_key=os.getenv("api_key"),
         base_url=os.getenv("base_url"),
     )
     responses = client.chat.completions.create(
         model="deepseek-r1",
         messages=[
             # {"role": "user", "content": "历史上的6月19发生了什么事情?"}
             {"role": "system", "content": "你是一个精通小说创作和鉴赏的小说作者。"},
             {"role": "user", "content": "我是社会精英，我住在江苏，我喜欢阅读"},
             {"role": "assistant", "content": "你好社会精英，希望你能体会阅读的快乐"},
             # {"role": "user", "content": "我是什么身份？我家住哪里？"},
             {"role": "user", "content" : "请描述下赘婿小说的艺术性和深度"}
         ],
         temperature=0,
         # top_p=0.95,
         max_tokens=500,
     )
     print(responses.choices[0].message.content)