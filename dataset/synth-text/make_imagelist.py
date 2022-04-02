import os

def file_name(file_dir):
    image_list=[]
    for root, dirs, files in os.walk(file_dir):
         #print(root) #当前目录路径
         #print(dirs) #当前路径下所有子目录
         #print(files) #当前路径下所有非目录子文件
         image_list.append(files)
    return image_list[0]


if __name__=="__main__":
    file='/home/uircv/桌面/cv/ocr/datasets/SynthText/SynthText/gt'
    image_list=file_name(file)
    print(image_list)
    filename='/home/uircv/桌面/cv/ocr/datasets/SynthText/SynthText/image_list.txt'

    with open(filename,'w',encoding="utf-8") as f:
        for name in image_list:
            f.write(name)
            f.write("\n")
