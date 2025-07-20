import jieba

if __name__ == '__main__':
    
    s = "生活用最残酷的方式让你笑，笑得哭出来。"
    result1 = jieba.lcut(s)
    # result2 = jieba.lcut_for_search(s)
    print(result1)
    # print(result2)