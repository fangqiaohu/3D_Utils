"""There is no download entrance to download 'ShapeNet' dataset in shapenet [https://www.shapenet.org].
This code is for crawling 'ShapeNet' dataset with multi-processing.
NOTE: The 'category' parameter below is for crawling a specific category.
To check the category, just right click the item and press 'Inspect' in your browser.
"""

# -*- coding:UTF-8 -*-
import requests
# from bs4 import BeautifulSoup
import multiprocessing
from urllib import request
import pandas as pd
import time
import os
from io import StringIO

# Select a category in shapenet.
# Change this to download your own category.
category = '02913152'  # building
# category = '02898711'  # bridge

# get query result in shapenet
target = '''https://www.shapenet.org/solr/models3d/select?q=wnhypersynsets%3A''' + category + '''&rows=10000&wt=csv'''
query_result = requests.get(url=target).text
ids_str = StringIO(query_result)
ids = list(pd.read_csv(ids_str, low_memory=False)['fullId'])

# # if already has a *.csv file:
# ids = list(pd.read_csv('ids.csv', low_memory=False)['fullId'])

ids = [id.split('.')[1] for id in ids]
ids.sort()


def save_model(id):

    progress = '%4s/%4s' % (ids.index(id)+1, len(ids))

    # # debug
    # id = 'b57d34cc0d45f29bb451d5c92b8ff593'

    url_kmz = '''https://www.shapenet.org/shapenet/data/%s/%s/%s/%s/%s/%s/%s/Collada/%s.kmz''' % (id[0], id[1], id[2], id[3], id[4], id[5:], id, id)
    url_img = '''https://www.shapenet.org/shapenet/data/%s/%s/%s/%s/%s/%s/%s/Image/%s''' % (id[0], id[1], id[2], id[3], id[4], id[5:], id, id)

    if not os.path.exists('models_%s/' % category):
        os.mkdir('models_%s/' % category)
    if not os.path.exists('images_%s/' % category):
        os.mkdir('images_%s/' % category)

    fn_write_kmz = 'models_%s/%s.kmz' % (category, id)
    fn_write_img = 'images_%s/%s.jpg' % (category, id)

    if os.path.exists(fn_write_kmz) and os.path.exists(fn_write_img):
        print('Exists! [%s] [%s].' % (progress, id))
        return 0

    try:
        request.urlretrieve(url=url_kmz, filename=fn_write_kmz)
        request.urlretrieve(url=url_img, filename=fn_write_img)
        print('Saved!  [%s] [%s].' % (progress, id))
        time.sleep(0.2)

    except:
        print('Failed! [%s] [%s].' % (progress, id))
        return 0


def multi_processing(iter):
    id = ids[iter]
    save_model(id)


# multiprocessing
if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=16)
    pool.map(multi_processing, range(len(ids)))
    pool.close()
    pool.join()
