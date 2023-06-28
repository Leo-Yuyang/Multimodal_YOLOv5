import os

root_path = '/home/ubuntu/dataset/kaist dataset/gy_copy/train/visible/'
txt_list = os.listdir(root_path)
# set06_V000_visible_I00899.txt
# set06_V000_I00899.txt
# set06_V000_I00459.png


# set06_V002_lwir_I01459.png
# set09_V000_lwir_I01679.png
# set06_V000_visible_I00019.png


for item in txt_list:
    item_new = item[:11]+'lwir_'+item[19:]
    # print(item)
    print(item_new)
    os.rename(root_path+item,root_path+item_new)



