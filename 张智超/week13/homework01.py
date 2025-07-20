import requests
import json

if __name__ == '__main__':
    url = 'http://localhost:11434/api/generate'
    req = {
        'model': 'deepseek-r1:1.5b',
        'prompt': '请推荐3本书籍，内容控制在100字以内。请直接给出最终答案，不需要解释或思考过程。',
        'stream': False
    }
    resp = requests.post(url, json=req)
    resp_json = json.loads(resp.text)
    print(resp_json['response'])