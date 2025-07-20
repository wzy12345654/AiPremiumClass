import os
import json
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI


class SmartLibraryAI:
    def __init__(self):
        load_dotenv(find_dotenv())
        self.client = OpenAI(
            api_key=os.getenv("api_key"),
            base_url=os.getenv("base_url", "https://api.openai.com/v1")
        )

        # 模拟数据库
        self.books_db = {
            "三体": {"author": "刘慈欣", "genre": "科幻", "stock": 3, "borrowers": []},
            "活着": {"author": "余华", "genre": "文学", "stock": 1, "borrowers": []},
            "Python编程": {"author": "Eric Matthes", "genre": "技术", "stock": 5, "borrowers": []},
            "百年孤独": {"author": "马尔克斯", "genre": "文学", "stock": 2, "borrowers": []},
            "银河帝国": {"author": "阿西莫夫", "genre": "科幻", "stock": 4, "borrowers": []}
        }

        # 用户借阅记录
        self.user_records = {
            "Lin老师": ["三体", "Python编程"]
        }

    def generate_prompt(self, user_input, user_id="default_user"):
        return f"""你是一个智能图书管理员，请严格按照以下规则处理请求：
{json.dumps(self.books_db, indent=2, ensure_ascii=False)}
{self.user_records.get(user_id, [])}
请用中文回复，且只返回需要直接对用户说的内容："""

    def process_request(self, user_input, user_id="default_user"):
        """处理用户请求的核心方法"""
        try:
            response = self.client.chat.completions.create(
                model="glm-4-flash-250414",
                messages=[
                    {"role": "system", "content": "你是一个专业的图书管理AI"},
                    {"role": "user", "content": self.generate_prompt(user_input, user_id)}
                ],
                temperature=0.3,
                max_tokens=500
            )

            result = response.choices[0].message.content
            self._update_db(user_input, user_id)  # 根据AI响应实际更新数据库
            return result

        except Exception as e:
            return f"系统错误：{str(e)}"

    def _update_db(self, user_input, user_id):
        """根据AI响应自动更新数据库（模拟实现）"""
        # 实际应用中应解析AI响应并准确更新
        # 这里简化为关键词触发更新
        if "借阅" in user_input:
            book = user_input.replace("借阅", "").strip()
            if book in self.books_db:
                self.books_db[book]["stock"] -= 1
                self.books_db[book]["borrowers"].append(user_id)
                if user_id not in self.user_records:
                    self.user_records[user_id] = []
                self.user_records[user_id].append(book)

        elif "归还" in user_input:
            book = user_input.replace("归还", "").strip()
            if book in self.books_db and user_id in self.books_db[book]["borrowers"]:
                self.books_db[book]["stock"] += 1
                self.books_db[book]["borrowers"].remove(user_id)
                self.user_records[user_id].remove(book)


if __name__ == "__main__":
    library_ai = SmartLibraryAI()

    # 测试用例
    test_cases = [
        ("Lin老师", "我想借阅三体"),
        ("Lin老师", "有什么科幻书推荐吗？"),
        ("Lin老师", "归还三体"),
        ("new_user", "我喜欢文学类书籍")
    ]

    for user_id, query in test_cases:
        print(f"\n[用户 {user_id}]: {query}")
        print(f"[AI回复]: {library_ai.process_request(query, user_id)}")
        print(f"当前库存: { {k: v['stock'] for k, v in library_ai.books_db.items()} }")