# coding=UTF-8

import pandas as pd
import sys
import time
# 这里是为了解决movielens 100k数据集的拼接问题



# test = pd.read_json('[{"a":2},{"a":3}]', orient="records")
# print(test.head())
# exit()

header = ["user_id", "movie_id", "rating", "timestamp",
          "title", "genres", "gender", "age", "occupation", "zip"]

rank_data = pd.read_csv("u.data", sep="\t", header=None)

# 修改列名
rank_data.columns = ["user_id", "movie_id", "rating", "timestamp"]
# print(rank_data.head(n=3))


item_data = pd.read_csv("./u.item", sep="|",
                        header=None, encoding="ISO-8859-1")
# print("info")
# item_data.info()
# print("head")
# print(item_data.head(3))

item_header_all = ["movie_id", "title", "release_date", "video release date",
            "IMDb URL", "unknown", "Action", "Adventure", "Animation",
              "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"]

genres_info = ["Action", "Adventure", "Animation",
              "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"]
# new_item_data = item_data.iloc[:, :3]
item_data.columns = item_header_all # 替换列名
# 由于训练用的数据需要获取分类名称信息，所以需要替换一下

def get_genres(row):
    genres = ""
    for i in genres_info:
        if row[i]==1:
            genres += "|" + i
    return genres[1:]

test_df = item_data.iloc[:,:].copy()
# print("test_df",test_df.head(3))

start = time.time()
test_df['genres'] = test_df.apply(lambda row: get_genres(row), axis=1)  # apply 好像快些

elapsed = (time.time() - start)
print("1 Time used:",elapsed)
# print(test_df.head(3))
# start = time.time()
# for i in range(len(test_df)):
#     row = test_df.iloc[i]
#     test_df.loc[i,'genres'] = get_genres(row)
# elapsed = (time.time() - start)
# print("2 Time used:",elapsed)


# 这里只需要  "movie_id", "title","genres"

item_info = test_df.loc[:,["movie_id", "title","genres"]]
# print(item_info.head(3))


# new_item_data = item_data.loc[:, :3]

# item_header = ["movie_id", "title", "release_date", "video release date"
#               "IMDb URL", "unknown", "Action", "Adventure", "Animation",
#               "Children's" "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
#               "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
#               "Thriller", "War", "Western"]

# print(new_item_data.head(n=3))
# pd.merge(rank_data, new_item_data, how="left", on="movie_id"）
# 开始merge
res1 = pd.merge(rank_data, item_info, how="left", on="movie_id")
# print(res.head(n=3))

# 用户信息表 ["user_id", "age", "gender", "occupation", "zip code"]


user_data = pd.read_csv("u.user", sep="|", header=None)

# 修改列名
user_data.columns = ["user_id", "age", "gender", "occupation", "zip"]
 
# print("user data:", user_data.head(3))

res2 = pd.merge(res1, user_data, how="left", on="user_id")
# print(res2.head(n=3))

res3 = res2.iloc[:,:].copy() # 复制一份
# import json
# print(test_res2.to_json(orient="records"))
res3 = res3[header] # 这里要调节一下列的顺序
# print("test_res info\n")
# test_res2.info()
# print("test_res info\n", test_res2.head())

res3.to_csv("hello.txt", header=1, index=0)