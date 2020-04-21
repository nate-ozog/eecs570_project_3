#!python3

import os

len_ = [100, 500, 1000, 2500, 5000, 7500, 10000]

for i  in range(len(len_)):
  for j in range(i+1):
    tlen = len_[i]
    qlen = len_[j]
    cmd_str = '../batchgen.o --tc 100 --qpt 100 --tl {} --ql {} --output T{}_Q{}.txt'.format(tlen, qlen, tlen, qlen)
    print('command: {}'.format(cmd_str))
    os.system(cmd_str)
