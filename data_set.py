#!/usr/bin/env python3
import sys
review_data = "ml-100k/u.data"
item_data = "ml-100k/u.item"

def load(train_file=None):
    """必要なデータをロードする"""
    if train_file:
        review_data = train_file
    user_dict = load_review_data(review_data)
    item_dict = load_item_data(item_data)
    return (user_dict, item_dict)

def load_review_data(file_path):
    with open(file_path) as f:
        lines = f.read().strip().split("\n")
        
    user_dict = {}
    for line in lines:
        usr_id, item_id, rating, timestmp = map(int,line.split())
        user_dict.setdefault(usr_id, {})
        user_dict[usr_id].setdefault(item_id, {})
        user_dict[usr_id][item_id] = rating

    return user_dict

def load_item_data(file_path):
    with open(file_path, encoding='latin-1') as f:
        lines = f.read().strip().split("\n")
        
    item_dict = {}
    items = []
    for line in lines:
        item_id, title, *_ = line.split("|")
        item_id = int(item_id)
        if item_id not in items: items.append(item_id)
        item_dict[item_id] = title

    return item_dict

