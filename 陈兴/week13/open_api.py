import openai
import os

def setup_api():
    """设置OpenAI API"""
    api_key = os.getenv('GLM_API_KEY')
    if not api_key:
        print("请设置环境变量: export GLM_API_KEY='your-api-key'")
        return None
    
    # OpenAI库的设置方式
    client = openai.OpenAI(api_key=api_key, base_url="https://open.bigmodel.cn/api/paas/v4/")
    return client

def test_temperature(client):
    """测试温度参数的影响, 控制生成文本的随机性，值越大，生成文本的随机性越大"""

    prompt = "写一个关于猫的短故事"
    
    print("=== 温度参数测试 ===")
    print(f"提示词: {prompt}")
    print()
    
    # 只测试两个温度值
    temperatures = [0.1, 1.0]
    
    for temp in temperatures:
        print(f"Temperature = {temp}:")
        try:
            response = client.chat.completions.create(
                model="glm-4", 
                messages=[{"role": "user", "content": prompt}],
                temperature=temp,
                max_tokens=100
            )
            print(f"输出: {response.choices[0].message.content}")
            print(f"Token使用: {response.usage.total_tokens}")
        except Exception as e:
            print(f"错误: {e}")
            print("提示: 请检查您的API密钥和网络连接")
        print("-" * 50)

def test_max_tokens(client):
    """测试最大token数的影响"""
    prompt = "解释什么是人工智能"
    
    print("=== 最大Token数测试 ===")
    print(f"提示词: {prompt}")
    print()
    
    # 只测试两个max_tokens值
    max_tokens_list = [50, 200]
    
    for max_tokens in max_tokens_list:
        print(f"Max Tokens = {max_tokens}:")
        try:
            response = client.chat.completions.create(
                model="glm-4", 
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=max_tokens
            )
            print(f"输出: {response.choices[0].message.content}")
            print(f"Token使用: {response.usage.total_tokens}")
        except Exception as e:
            print(f"错误: {e}")
            print("提示: 请检查您的API密钥和网络连接")
        print("-" * 50)

def main():
    """主函数"""
    print("OpenAI API 参数调试工具")
    print("=" * 30)
    print()
    
    client = setup_api()
    if not client:
        return
    
    # 测试温度参数
    test_temperature(client)
    
    print()
    
    # 测试最大token数
    test_max_tokens(client)
    
    print("\n调试完成!")

if __name__ == "__main__":
    main()
