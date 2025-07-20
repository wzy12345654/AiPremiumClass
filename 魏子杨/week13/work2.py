import os
from openai import OpenAI
from dotenv import find_dotenv, load_dotenv
if __name__=='__main__': 
 load_dotenv(find_dotenv()) #find_dotenv()找到当前项目中的env文件的路径  load_dotenv加载env

 client = OpenAI(
    api_key=os.getenv('API_KEY'), # 必须指定，⽤于⾝份验证
    base_url=os.getenv('BASE_URL'), # 提供模型调⽤的服务URL
 )
response = client.chat.completions.create(
 	model="glm-4-flash-250414", #大模型名称
 	messages=[
 		{"role": "user", "content": "请告诉我谁是世界上最美的女人"}
 	]
 )
print(response.choices[0].message.content)