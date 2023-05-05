import os
import shutil

# dir = "FCA/"
root_path = "/home/ubuntu/dataset/kaist_pixel_level/preparing_data/test/RGB/"
to_path = "/media/ubuntu/IEEEARM2021/copy/result/small_data/saliency map/"+"image/"

# save_list = ['set06_V000_I01099.jpg_pred.jpg','set06_V002_I01219.jpg_pred.jpg','set06_V003_I00659.jpg_pred.jpg','set06_V003_I02959.jpg_pred.jpg','set08_V000_I00379.jpg_pred.jpg','set08_V000_I02159.jpg_pred.jpg','set08_V002_I00859.jpg_pred.jpg','set10_V000_I03599.jpg_pred.jpg']
# save_list = ['set06_V000_I01099.jpg_labels.jpg','set06_V002_I01219.jpg_labels.jpg','set06_V003_I00659.jpg_labels.jpg','set06_V003_I02959.jpg_labels.jpg','set08_V000_I00379.jpg_labels.jpg','set08_V000_I02159.jpg_labels.jpg','set08_V002_I00859.jpg_labels.jpg','set10_V000_I03599.jpg_labels.jpg']
save_list = ['set06_V003_I01439.png','set06_V003_I02799.png','set10_V000_I00139.png','set11_V000_I01399.png']

img_list = os.listdir(root_path)

for i in img_list:
    if i in save_list:
        print(i)
        shutil.copy(root_path+i,to_path+"RGB"+i)












