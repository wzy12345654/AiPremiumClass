import os
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv(find_dotenv())

    client = OpenAI(
        api_key=os.environ["api_key"],
        base_url=os.environ["base_url"]
    )

    response = client.chat.completions.create(
        model="glm-z1-flash",
        messages=[
            {'role':'user','content':'列举一些中国的传统节日'},
        ],
        temperature=0.7,
        top_p=0.9,
        max_tokens=100
    )

    print(response.choices[0].message.content)
