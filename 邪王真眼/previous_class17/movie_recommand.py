import os
import heapq
import openpyxl
import numpy as np


def build_u2i_matrix(data_path, write_file=False):
    item_id_to_item_name = {}
    with open(os.path.join(data_path, 'u.item'), encoding="ISO-8859-1") as f:
        for line in f:
            item_id, item_name = line.split("|")[:2]
            item_id = int(item_id)
            if item_id > 500:
                continue
            item_id_to_item_name[item_id] = item_name

    total_movie_count = len(item_id_to_item_name)
    print("total movie:", total_movie_count)

    user_to_rating = {}
    with open(os.path.join(data_path, 'u.data'), encoding="ISO-8859-1") as f:
        for line in f:
            user_id, item_id, score, _ = line.split("\t")
            user_id, item_id, score = int(user_id), int(item_id), int(score)
            if user_id > 200 or item_id > 500:
                continue
            if user_id not in user_to_rating:
                user_to_rating[user_id] = [0] * total_movie_count
            user_to_rating[user_id][item_id - 1] = score
    print("total user:", len(user_to_rating))

    if not write_file:
        return user_to_rating, item_id_to_item_name

    workbook = openpyxl.Workbook()
    sheet = workbook.create_sheet(index=0)
    
    header = ["user_id"] + [item_id_to_item_name[i + 1] for i in range(total_movie_count)]
    sheet.append(header)
    for i in range(len(user_to_rating)):
        line = [i + 1] + user_to_rating[i + 1]
        sheet.append(line)
    workbook.save("./邪王真眼/previous_class17/result/user_movie_rating.xlsx")
    return user_to_rating, item_id_to_item_name


def user_cf(user_id, item_id, user_to_similar_user, user_to_rating, topn=10):
    pred_score = 0
    count = 0
    for similar_user, similarity in user_to_similar_user[user_id][:topn]:
        rating_by_similiar_user = user_to_rating[similar_user][item_id - 1]
        pred_score += rating_by_similiar_user * similarity
        if rating_by_similiar_user != 0:
            count += 1
    pred_score /= count + 1e-5
    return pred_score


def item_cf(user_id, item_id, similar_items, user_to_rating, topn=10):
    pred_score = 0
    count = 0
    for id, similarity in similar_items[item_id]:
        if id == item_id:
            pred_score += similarity
            count += 1

    pred_score /= count + 1e-5
    return pred_score


def movie_recommand(user_id, similar_user, similar_items, user_to_rating, item_to_name, topn=10):
    unseen_items = [item_id + 1 for item_id, rating in enumerate(user_to_rating[user_id]) if rating == 0]
    res = []
    for item_id in unseen_items:
        # score = user_cf(user_id, item_id, similar_user, user_to_rating)
        score = item_cf(user_id, item_id, similar_items, user_to_rating)
        res.append([item_to_name[item_id], score])
    
    res = sorted(res, key=lambda x:x[1], reverse=True)
    return res[:topn]


def cosine_distance(vector1, vector2):
    ab = vector1.dot(vector2)
    a_norm = np.sqrt(np.sum(np.square(vector1)))
    b_norm = np.sqrt(np.sum(np.square(vector2)))
    return ab / (a_norm * b_norm + 1e-8)


def find_similar_user(user_to_rating):
    user_to_similar_user = {}
    user_to_rating = [
        (user, np.array(ratings))
        for user, ratings in user_to_rating.items()
    ]
    
    for user_a, ratings_a in user_to_rating:
        similar_user = []
        for user_b, ratings_b in user_to_rating:
            if user_a >= user_b:
                continue
            similarity = 1 - cosine_distance(np.array(ratings_a), np.array(ratings_b))

            if len(similar_user) < 10:
                heapq.heappush(similar_user, (similarity, user_b))
            else:
                if similarity > similar_user[0][0]:
                    heapq.heappushpop(similar_user, (similarity, user_b))

        top_users_sorted = sorted(similar_user, key=lambda x: -x[0])
        user_to_similar_user[user_a] = [(user, sim) for sim, user in top_users_sorted]
    return user_to_similar_user


def find_similar_item(user_to_rating):
    item_to_vector = {}
    total_user = len(user_to_rating)
    for user, user_rating in user_to_rating.items():
        for moive_id, score in enumerate(user_rating):
            moive_id += 1
            if moive_id not in item_to_vector:
                item_to_vector[moive_id] = [0] * (total_user + 1)
            item_to_vector[moive_id][user] = score
    return find_similar_user(item_to_vector)


if __name__ == "__main__":
    data_path = './邪王真眼/datasets/ml100k/ml-100k'
    user_to_rating, item_to_name = build_u2i_matrix(data_path, False)

    similar_user = find_similar_user(user_to_rating)
    similar_items = find_similar_item(user_to_rating)

    while True:
        user_id = int(input("输入用户id："))
        recommands = movie_recommand(user_id, similar_user, similar_items, user_to_rating, item_to_name)
        for recommand, score in recommands:
            print("%.4f\t%s"%(score, recommand))
