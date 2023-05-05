import os


root_path = '/home/ubuntu/dataset/kaist dataset/gy_copy/train/'
f1 = open(root_path+'train.txt','r')

f2 = open(root_path+'train_visible.txt','w')

# root = '/home/neu-wang/gongyan/big_data/'

f1_list = f1.read().splitlines()
for item in f1_list:
    # print(item)
    item = item.replace('thermal','visible')
    # print(item)
    # item_split = item.split('/')
    # item_1 = item_split[-4]+'/'+item_split[-3]+'/'+item_split[-2]+'/'+item_split[-1]
    # item_2 = root + item_1
    f2.write(item)
    f2.write('\n')
    # print(item_2)

f1.close()
f2.close()








