import os, json


dirname = os.path.dirname(os.path.abspath(__name__))

with open(os.path.join(dirname, 'valsketch/static/sketches_svg/filelist.txt')) as flist_file:
    file_list = map( lambda x: x[:-1], flist_file.readlines())

with open(os.path.join(dirname, 'valsketch/static/parts.json')) as data_file:
    part_list = json.load(data_file)
