from PIL import Image
import glob
img_list = glob.glob('/home/ubuntu/dataset/kaist_pixel_level/preparing_data/train/image/*.jpg')

for item in img_list:
    img = Image.open(item)
    # print(img.size[0],img.size[1])
    if img.size[0] !=640 or img.size[1] !=512:
        print("hah")






