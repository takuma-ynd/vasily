#!/usr/bin/env python3
"""Recommendation system
model description: http://yifanhu.net/PUB/cf.pdf
"""
import os
import argparse
import pickle
import ipdb
import numpy as np
import data_set

# random seedを固定
np.random.seed(1)

# 実行時の引数に関する設定
parser = argparse.ArgumentParser()
parser.add_argument("--train", help="training mode", action="store_true")
parser.add_argument("--eval", help="evaluation mode", action="store_true")
parser.add_argument("--train-file", help="data file")
parser.add_argument("--eval-file", help="evaluation file")
parser.add_argument("--model-file", help="model file(valid in evaluation mode)")
parser.add_argument("-f", help="hyper parameter: dimension of vectors",
                    type=int, default=50)
parser.add_argument("-a", help="hyper parameter: aplha (in calc of c)",
                    type=int, default=40)
parser.add_argument("--lmbda", help="hyper parameter: lambda",
                    type=float, default=1.0)
parser.add_argument("--num-epochs", help="hyper parameter: number of parameters",
                    type=int, default=15)
parser.add_argument("--log-file", help="log file for automatic training.")
args = parser.parse_args()

# hyper parametersをここにまとめておく
hyper_params = {
    "f": args.f, # dimension of vectors
    "a": args.a, # alpha
    "lmbda" : args.lmbda,
    "num_epochs" : args.num_epochs
}

def train_step(Y):
    """trainingの1stepに対応する処理を行う.
    new_X, new_Yは共に現状のYから計算できるため，引数はYのみ．
    """

    # new_x[u]を全ユーザについて計算
    new_x = []
    YtY = Y.T @ Y
    eye = np.eye(f)
    for u in range(m):
        if u % 50 == 0:
            print("calculating user {}/{}".format(u, m))
        Cu = np.diag(c[u])
        pu = p[u]
        new_x.append(
            np.linalg.inv(YtY + Y.T @ (Cu - np.eye(n)) @ Y + lmbda * eye) @ Y.T @ Cu @ pu
        )

    # new_y[i]を全アイテムについて計算
    new_y = []
    X = np.array(new_x)
    XtX = X.T @ X
    for i in range(n):
        if i % 50 == 0:
            print("calculating item {}/{}".format(i, n))
        Ci = np.diag(c.T[i])
        pi = p.T[i]
        new_y.append(
            np.linalg.inv(XtX + X.T @ (Ci - np.eye(m)) @ X + lmbda * eye) @ X.T @ Ci @ pi
        )
    Y = np.array(new_y)
    return (X, Y)

def recommend(specified_items=None, usr_id=None, review_dict=None, top_k=None):
    """入力されたユーザに対してtop_k個のrecommendation itemsを返す.

    引数:
    specified_items: recommendするitemを指定してlistで渡す．(Noneなら全てのitemから選択)
    usr_id: 対象とするユーザーのid．
    review_dict: ユーザーidを指定しない時，ユーザーを表現するために{item_id: review}なる辞書を渡す．
    top_k: recommendするitem数を指定する(default: 全itemを返す)

    返り値:
    topk_items: 推薦度が高い順にtop_k個のitem idをlist形式で返す

    式に対応する各変数名は論文における式中の各文字に準じている．
    """

    refY = Y # Yへの参照
    if specified_items is None:
        specified_items = range(1, n+1)
        
    if usr_id:
        # userに対応するベクトルをlookupしてpreferenceを計算
        user_vect = X[usr_id-1]
        preferences = refY @ user_vect
        topk_items = (-preferences).argsort()[:top_k] + 1 # idに直すため，+1が必須

    elif review_dict:
        # 各変数名は論文に登場する文字に準じている．
        YtY = refY.T @ refY
        r_u = np.array([review_dict.get(i, 0) for i in range(1, n+1)])
        c_u = [1 if e ==0 else a for e in r_u]
        # c_u = 1 + a*r_u
        Cu = np.diag(c_u)
        Wu = np.linalg.inv(YtY + refY.T @ (Cu - np.eyes(n)) @ refY + lmbda * np.eyes(f))
        
        pu = {i:refY[i-1] @ Wu @ refY.T @ c_u for i in range(1, n+1)}
        item_score_pairs = sorted(pu.items(), key=lambda x:x[1], reversed=True)
        topk_items = [item_id for item_id, score in item_score_pairs if item_id in specified_items][:top_k]

    else:
        raise RuntimeError("specify a user: input usr_id or usr_dict")
    
    return topk_items

def evaluate(file_name):
    """expected percentile rankingを計算して返す.
    引数:
    file_name: evaluationに用いるテスト用ファイル(訓練用データと同じ形式)
    
    返り値:
    mean_rank: 論文参照．(乱数でパラメータ初期化後に学習しなければ0.5．低いほどよい．)
    """
    user_dict = data_set.load_review_data(file_name)

    # 分母を計算する．
    denominator = 0
    item_ids = []
    for user_id, items in user_dict.items():
        for item_id, review in items.items():
            denominator += review
            if item_id not in item_ids: item_ids.append(item_id)

    # 分子を計算する．
    numerator = 0
    for user_id, items in user_dict.items():
        # user_dictに含まれるitem(item_ids)のみをrecommendの候補とする．
        recommended_items = recommend(specified_items=item_ids, usr_id=user_id)
        num_items = len(recommended_items)
        for item_id, review in items.items():
            # recommended items中の，item_idがあるindexを返す．
            np_idx = np.where(recommended_items == item_id)
            assert len(np_idx) == 1
            idx = np_idx[0][0]
            rank_ui = idx / num_items
            numerator += review * rank_ui

    mean_rank = numerator / denominator
    return mean_rank

def save(save_dict, file_name):
    """save_dictをpickle形式で保存する．"""
    if os.path.exists(file_name):
        raise OSError("file: {} already exists.".format(file_name))

    saver = {}
    for key, val in save_dict.items():
        saver[key] = val
        
    with open(file_name, "wb") as f:
        pickle.dump(saver, f)
    print("data saved in {}".format(file_name))

def load(file_name):
    """file_nameで指定されたpickleファイルを読み込む"""
    with open(file_name, "rb") as f:
        loader = pickle.load(f)

    return loader


if __name__ == "__main__":

    # 評価
    if args.eval:
        saved = load(args.model_file)
        locals().update(saved) # savedに含まれる辞書でローカル変数を更新する．
        mean_rank = evaluate(args.eval_file)
        print("mean_rank: {}".format(mean_rank))

    # 訓練
    elif args.train:
        # ハイパーパラメータ
        f = hyper_params["f"]
        a = hyper_params["a"]
        lmbda = hyper_params["lmbda"]
        num_epochs = hyper_params["num_epochs"]

        user_dict, item_dict = data_set.load(args.train_file)

        n = len(item_dict)
        m = len(user_dict)

        # データを読み込んでX,Yに格納
        # X, Yの定義及び初期化
        X = np.random.rand(m, f)
        Y = np.random.rand(n, f)

        # r, c, p の定義
        r = np.array([[user_dict[u].get(i, 0) for i in range(1,n+1)] for u in range(1,m+1)])
        c = np.array([[1 if e == 0 else a for e in r_row] for r_row in r]) # confidence
        # p = (r != 0).astype(float)
        p = np.array([[1 if e > 2 else 0 for e in r_row] for r_row in r]) # preference

        # 学習
        for epoch in range(num_epochs):
            print("---------- epoch {}/{} ----------".format(epoch, num_epochs))
            X, Y = train_step(Y)

        if args.model_file:
            # save_typesで指定した型のlocal変数を全て保存
            save_types = [int, float, np.ndarray, dict]
            local_vars = {key:val for key, val in locals().items() if not key.startswith("_") and type(val) in save_types}
            save(local_vars, args.model_file)

        if args.eval_file: # evaluation fileを指定している場合はevaluationを行う．
            mean_rank = evaluate(args.eval_file)
            print("f:{}\ta:{}\tlmbda:{}\tnum_epochs:{}\tmean_rank:{}\n".format(f, a, lmbda, num_epochs, mean_rank))
            print("mean_rank: {}".format(mean_rank))
        else:
            mean_rank = None
        
        if args.log_file: # log fileを指定している場合はログを取る．
            out = "f:{}\ta:{}\tlmbda:{}\tnum_epochs:{}\tmean_rank:{}\n".format(f, a, lmbda, num_epochs, mean_rank)
            with open(args.log_file, "a") as lf:
                lf.write(out)
            
    else:
        print("specify the mode.\n use -h option to see hints.")
