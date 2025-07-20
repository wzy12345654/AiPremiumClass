
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

if __name__ == '__main__':
    load_dotenv()
    api_key = os.getenv('API_KEY')
    base_url = os.getenv('BASE_URL')
    model = ChatOpenAI(
        api_key=api_key,
        base_url=base_url,
        model='glm-4-plus',
        temperature=0.9
    )
    tpl = ChatPromptTemplate.from_messages([
        ('system', """你是一个图书馆管理员，为读者提供服务：借阅图书、归还图书、查询当前在库的图书。如果读者想借阅图书，流程如下：
        首先查询库存中还有没有该图书，如果没有则回复无法借阅以及原因，如果有则需要读者输入图书ID。你只需要一句话告诉读者借阅归还成功与否以及原因，同时更新图书库存信息。
        现在库存有以下这几本图书：

一、经典文学‌
001.《百年孤独》‌（加西亚·马尔克斯）剩余1本
魔幻现实主义巅峰之作，布恩迪亚家族七代人的命运轮回，揭示孤独与宿命的永恒命题。
002.《你当像鸟飞往你的山》‌（塔拉·韦斯特弗）剩余2本
自传体小说，讲述作者冲破极端原生家庭束缚，通过教育重塑自我的震撼历程。
003.《活着》‌（余华）剩余3本
福贵历经战争与亲人离世的苦难一生，展现生命在绝望中的坚韧与尊严。
004.《杀死一只知更鸟》‌（哈珀·李）剩余1本
通过种族审判事件，拷问人性偏见与勇气的本质，传递“理解他人”的普世价值。
二、思想启蒙‌
005.《1984》‌（乔治·奥威尔）剩余1本
反乌托邦经典，刻画极权社会下的思想控制，警醒自由与真相的珍贵。
006.《思考，快与慢》‌（丹尼尔·卡尼曼）剩余1本
诺贝尔奖得主剖析人类决策机制，揭示认知偏见与理性陷阱。
007.《人类简史》‌（尤瓦尔·赫拉利）剩余1本
颠覆性视角重构人类文明史，追问科技革命下的幸福与未来。
三、心理成长‌
008.《被讨厌的勇气》‌（岸见一郎 & 古贺史健）剩余1本
以阿德勒心理学破解人际关系困局，倡导“课题分离”的自由生活哲学。
四、社会纪实‌
009.《我的母亲做保洁》‌剩余1本
双视角记录城市底层与中产阶层的生存困境，折射城市化进程中的身份焦虑。
010.《格外的活法》‌（吉井忍）剩余1本
日籍作家重返东京的观察手记，探索格子社会外的个体生存可能性（2025年新作）。
"""),
        MessagesPlaceholder(variable_name='messages')
    ])
    outputParser = StrOutputParser()

    def count_token(messages):
        return len(messages)
    trimer = trim_messages(max_tokens=200, strategy='last',
                           token_counter=count_token)
    chain = tpl | trimer | model | outputParser
    session_store = {}

    def get_session_history(session_id):
        if session_id not in session_store:
            session_store[session_id] = InMemoryChatMessageHistory()
        return session_store[session_id]
    runnable = RunnableWithMessageHistory(
        chain, get_session_history, input_messages_key='messages')
    session_id = 123

    while True:
        in_msg = input("<<< ")
        if not in_msg:
            break
        if in_msg == 'N':
            session_id += 1
            continue
        response = runnable.invoke({
            "messages": [
                HumanMessage(in_msg)
            ]
        }, config={
            'configurable': {
                'session_id': session_id
            }
        })
        print('>>> ', response)
