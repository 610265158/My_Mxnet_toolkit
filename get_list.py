# -*- coding:utf-8 -*-

import os
import random

def main(ratio):
    workdir=os.getcwd()
    datadir=os.path.join(workdir,'data')
    name_list=os.listdir(datadir)

    ratio=ratio

    train_list=open('train.lst',mode="w+", encoding='utf-8');
    val_list=open('val.lst',mode="w+", encoding='utf-8');

    count=0
    count_val=0

    for name in name_list:
        pics_dir=os.path.join(datadir,name)
        pic_list=os.listdir(pics_dir)

        random.shuffle(pic_list)
        train_l=pic_list[0:int(len(pic_list)*ratio)]
        val_l = pic_list[int(len(pic_list) * ratio):]
        for pic in train_l:

            tmp_string=str(count)+'\t'+str((name_list.index(name)))+'\t'+pics_dir+'/'+pic +'\n'
            train_list.write(tmp_string)
            count+=1
        for pic in val_l:
            tmp_string = str(count_val) + '\t' + str((name_list.index(name))) + '\t' +pics_dir+'/'+pic +'\n'
            val_list.write(tmp_string)
            count_val+=1

    train_list.close()
    val_list.close()

    for i in range(len(name_list)):
        print(name_list[i],'with label',i)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process train and val list.')
    parser.add_argument('--ratio', dest='ratio',type=float, default=0.8,  help='the ratio between train an val (default: 0.8)')
    args = parser.parse_args()

    main(args.ratio)