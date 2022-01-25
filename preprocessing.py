import pandas as pd
import os
import SimpleITK
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import cv2
from sklearn.model_selection import train_test_split
def preprocess_data_images(root,csv_file):
  execute_in_docker = True    
  data = pd.read_csv(csv_file)
  for counter,image in enumerate(os.listdir(os.path.join(root,'images'))):
    print(counter)
    img_path = os.path.join(root, "images",str(image))
    img = SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(img_path))
    img = np.around(img,decimals = 0)
    img = np.asarray(img, dtype = 'uint16')
    img = Image.fromarray(img)
    image_name = str(image)
    sub = image_name.split('.')
    im_name = sub[0]
    boxes1 = []
    if im_name[0]=='n':
      print('annotations being saved',counter)
      img_path2 = 'darknet/data/obj/' if execute_in_docker else './darknet/data/obj/'
      nodule_data = data[data['img_name']==str(image)]
      print('nodule_data',nodule_data)
      num_objs = len(nodule_data)
      print('num_objs',num_objs)
      '''if im_name[0]=='n':'''
      for i in range(num_objs):
        x = nodule_data.iloc[i]['x']
        y =  nodule_data.iloc[i]['y']
        w = nodule_data.iloc[i]['width']
        h = nodule_data.iloc[i]['height']
        boxes1.append([x, y, w, h])
        save_path = 'darknet/data/annotations' if execute_in_docker else './darknet/data/annotations'
        save_path2 = 'darknet/data/obj' if execute_in_docker else './darknet/data/obj'
        file_name2 = im_name+'.txt'
        comp_name = os.path.join(save_path,file_name2)
        num = 1024
        comp_name2 = os.path.join(save_path2,file_name2)
        with open (comp_name,'w') as f:
          newline = str('0')+' '+str(x/num)+' '+ str(y/num) +' ' + str(w/num)+ ' ' +str(h/num)
          f.write(newline)
        with open (comp_name2,'w') as f1:
          newline = str('0')+' '+str(x/num)+' '+ str(y/num) +' ' + str(w/num)+ ' ' +str(h/num)
          f1.write(newline) 
    else :
        img_path2 = 'darknet/data/images_no_nodules/' if execute_in_docker else './darknet/data/imags_no_nodules'
        
    image_name = im_name+'.png'
    #image_namelist.append(image_name)
    #brighttt = ImageEnhance.Brightness(img)
    #out=brighttt.enhance(15)
    img.save(img_path2+im_name+'.png')
    img_path3 = os.path.sep.join([img_path2,image_name])
    img1 = cv2.imread(img_path3)
    img = Image.fromarray(img1)
    print('img resizing',counter)
    newsize = (608,608)
    img = img.resize(newsize)
    #print(im.dtype)
    brighttt = ImageEnhance.Brightness(img)
    out=brighttt.enhance(15)
    #save_path ='/content/darknet/data/obj/
    print('image getting saved',counter)   
    out.save(img_path2+image_name)
  return

def preprocess(img,idx):
  execute_in_docker = True    
  print('1',img.shape)
  img = np.around(img,decimals = 0)
  img = np.asarray(img, dtype = 'uint16')
  img = Image.fromarray(img)
  #print('2',img.shape)
  
  im_name = str(idx)
  image_name = im_name+'.png'
  save_path = ''
  file_name = 'test1'
  img_path2 = 'test1/' if execute_in_docker else './test1'  
  img.save(img_path2+im_name+'.png')
  img_path3 = os.path.sep.join([img_path2,image_name])
  print(img_path3)
  img1 = cv2.imread(img_path3)
  img = Image.fromarray(img1)
  
  newsize = (608,608)
  print('resizing img',counter)
  img = img.resize(newsize)
  #print('aftr resizing',img.shape)
  #print(im.dtype)
  brighttt = ImageEnhance.Brightness(img)
  out=brighttt.enhance(15)
  #save_path ='/content/darknet/data/obj/
  print('image getting saved')   
  out.save(img_path2+image_name)
  return

def splitting():
    execute_in_docker = True
    save_path = 'darknet/data' if execute_in_docker else './darknet/data'
    file_name1 = 'train.txt'
    file_name2 = 'test.txt'
    name1 = os.path.join(save_path,file_name1)
    name2 = os.path.join(save_path,file_name2)
    file_train= open(name1,'w')
    file_val = open('test.txt','w')
    if execute_in_docker:
        images = [os.path.join('darknet/data/annotations',x) for x in os.listdir('darknet/data/obj')]
    else:
        images = [os.path.join('./darknet/data/annotations',x) for x in os.listdir('./darknet/data/obj')]
    train_images,val_images = train_test_split(images,test_size = .2, random_state=1)
    for f in train_images:
        f1 = f.replace('annotations','obj')
        name = f1.split('.')
        file_train.write(name[0]+'.png'+'\n')
    for f in val_images:
        f1 = f.replace('annotations','obj')
        name = f1.split('.')
        file_val.write(name[0]+'.png'+'\n') 
    return    