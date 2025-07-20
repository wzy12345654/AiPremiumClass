from dotenv import find_dotenv, load_dotenv
import os
from langchain_openai import ChatOpenAI # LLM调用封装
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage # 对话角色：user、assistant、system
from langchain_core.output_parsers import StrOutputParser

#导入必要包
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# 导入template中传递聊天历史信息的“占位”类
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


if __name__ == '__main__':
    load_dotenv(find_dotenv())

    # Initialize the OpenAI Chat model
    model = ChatOpenAI(
        model="glm-4-flash-250414",
        base_url=os.environ['base_url'],
        api_key=os.environ['api_key'],
        temperature=0.7
    )
 
    # 带有占位符的prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            # ("system", "你是一个专业、友好、耐心的图书馆AI助手，致力于提供高效、个性化的图书借阅服务。你需要完成以下任务：图书借阅管理：接受用户输入的图书名称，确认是否可借。如果图书可借，完成借阅流程，并记录用户借阅信息。如果图书不可借，告知用户预计归还时间或建议类似书籍。图书归还管理：接收用户归还请求，更新库存信息。如果用户归还逾期，提醒归还时间与相关规则。个性化图书推荐：通过用户历史借阅记录或兴趣偏好，为其推荐相关书籍。可根据类型（如小说、科技、历史等）进行个性化推荐。用户：我想借《三体》。AI：好的，《三体》目前可借，我已经为您预留，请在三天内到馆领取。还需要我推荐其他科幻书籍吗？用户：帮我推荐几本小说类的书。AI：当然，您可能会喜欢《活着》《平凡的世界》《百年孤独》这几本高评分小说。用户：我要还书《活着》。AI：收到，已经为您处理归还。下次欢迎继续借阅！如果用户未提供偏好，请引导其说明喜欢的类型或之前看过的书。如果用户提供具体书名但该书被借出，优先推荐相同类别、作者或标签的其他可借图书。记录用户最近借阅记录，进行个性化推荐。"),
            # ("system", "接受用户输入的图书名称，确认是否可借。如果图书可借，完成借阅流程，并记录用户借阅信息。如果图书不可借，告知用户预计归还时间或建议类似书籍。图书归还管理：接收用户归还请求，更新库存信息。如果用户归还逾期，提醒归还时间与相关规则。个性化图书推荐：通过用户历史借阅记录或兴趣偏好，为其推荐相关书籍。可根据类型（如小说、科技、历史等）进行个性化推荐。") ,
            # ("system", "示例对话：用户：我想借《三体》。AI：好的，《三体》目前可借，我已经为您预留，请在三天内到馆领取。还需要我推荐其他科幻书籍吗？用户：帮我推荐几本小说类的书。AI：当然，您可能会喜欢《活着》《平凡的世界》《百年孤独》这几本高评分小说。用户：我要还书《活着》。AI：收到，已经为您处理归还。下次欢迎继续借阅！如果用户未提供偏好，请引导其说明喜欢的类型或之前看过的书。如果用户提供具体书名但该书被借出，优先推荐相同类别、作者或标签的其他可借图书。记录用户最近借阅记录，进行个性化推荐。") ,
            # ("system", "如果用户未提供偏好，请引导其说明喜欢的类型或之前看过的书。如果用户提供具体书名但该书被借出，优先推荐相同类别、作者或标签的其他可借图书。记录用户最近借阅记录，进行个性化推荐。") ,
            ("system", "你是一个专业且智能的管理图书馆图书的系统。你需要热情友好地询问读者需要的帮助。你的主要为读者提供服务：借阅图书，归还图书，查询当前在库的图书。你只需要一句话告诉读者借阅归还成功或者失败以及原因，后台更新的库存数量不需要展示。") ,

            ("system", "如果读者想查询图书，可以展示图书的书名和在库数量。") ,

            ("system", "如果读者想借阅图书，流程如下：首先查询文件里是否有该图书，如果没有或者库存为0则回复读者无法借阅以及原因；") ,
            ("system", "如果可以借阅该图书，则需要读者输入读书ID号；") ,
            ("system", "借阅成功后台记录借阅读者号ID更新库存数量减1。") ,

            ("system", "当读者想要归还某本图书，流程如下：确认需要归还的图书和读者的读书ID号；若确认图书为该读者借阅，完成归还流程。后台更新库存数量加1。若不是借阅的图书的读者归还时则提示读者，同时不能归还。") ,
            ("system", "完成一次借阅或者归还信息后，若再次需要借阅或者归还，需要重新输入读者ID号。") ,
            ("system", "图书信息在csv格式文件中：") ,
            ("system", "书名,作者,图书简介,类型,出版社,库存数量") ,
            ("system", "《星辰与海洋的秘密》,王小明,在这本书中，主人公通过一系列探险揭开星辰和海洋之间的神秘联系。,科幻,星际出版社,50") ,
            ("system", "《时光里的画师》,李华,这是一部关于一位才华横溢的画师如何在不同时空中追寻自己梦想的作品。,历史/传记,艺术之光出版社,32") ,
            ("system", "《未来之城》,张强,本书探讨了未来的城市规划和设计理念，以及技术进步对未来社会的影响。,建筑与科技,未来之梦出版社,100") ,
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    parser = StrOutputParser()
    # chain 构建
    chain =  prompt | model | parser
    
    # 定制存储消息的dict
    # key:  sessionId会话Id（资源编号）区分不同用户或不同聊天内容
    # value: InMemoryChatMessageHistory存储聊天信息list对象
    store = {}

    # 定义函数：根据sessionId获取聊天历史（callback 回调）
    # callback 系统调用时被执行代码
    def get_session_hist(session_id):
        # 以sessionid为key从store中提取关联聊天历史对象
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory() # 创建
        return store[session_id] # 检索 （langchain负责维护内部聊天历史信息）

    # 在chain中注入聊天历史消息
    # 调用chain之前，还需要根据sessionID提取不同的聊天历史
    with_msg_hist = RunnableWithMessageHistory(
        chain, 
        get_session_history=get_session_hist,
        input_messages_key="messages")  # input_messages_key 指明用户消息使用key
    
    # session_id
    session_id = "8888"

    while True:
        # 用户输入
        user_input = input('用户输入的Message：')
        # 调用注入聊天历史的对象
        response = with_msg_hist.invoke(
            {
                "messages":[HumanMessage(content=user_input)],
            },
            config={'configurable':{'session_id': session_id}} # 将session_id注入到chain中
        )
        print('AI Message:', response)



