import cv2
import os
from image_processing import func
if not os.path.exists("temp"):
    os.makedirs("temp")
if not os.path.exists("temp/train"):
    os.makedirs("temp/train")
if not os.path.exists("temp/test"):
    os.makedirs("temp/test")
path="data3/train"  #coloured images here
path1 = "temp"  #black and white images stored here



label=0 #number of characters
var = 0 #total number of images
c1 = 0 #total images in train
c2 = 0	#number images in test

for (dirpath,dirnames,filenames) in os.walk(path):
    for dirname in dirnames:
        print(dirname)
        for(direcpath,direcnames,files) in os.walk(path+"/"+dirname):
       	    if not os.path.exists(path1+"/train/"+dirname):
                os.makedirs(path1+"/train/"+dirname)
            if not os.path.exists(path1+"/test/"+dirname):
                os.makedirs(path1+"/test/"+dirname)
            num=0.8*len(files)
            #num = 100000000000000000
            i=0
            for file in files:
                var+=1
                actual_path=path+"/"+dirname+"/"+file
                actual_path1=path1+"/"+"train/"+dirname+"/"+file
                actual_path2=path1+"/"+"test/"+dirname+"/"+file
                img = cv2.imread(actual_path, 0)
                bw_image = func(actual_path)
                if i<num:
                    c1 += 1
                    cv2.imwrite(actual_path1 , bw_image)
                else:
                    c2 += 1
                    cv2.imwrite(actual_path2 , bw_image)
                    
                i=i+1
                
        label=label+1
print(var)
print(c1)
print(c2)
print(label)




