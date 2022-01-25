import os
def create_file_cfg(docker):
  execute_in_docker = True   
  save_path = '/opt/algorithm/darknet/cfg' if execute_in_docker else './darknet/cfg'
  file_name = 'yolov4-obj.cfg'
  comp_name = os.path.join(save_path,file_name)
  name1 = '/opt/algorithm/darknet/cfg/yolov4-custom.cfg' if execute_in_docker else './darknet/cfg'
  with open(name1,'r') as f1 ,open (comp_name,'a') as f:
    for line in f1:
      f.write(line)

  cfg_file = open(comp_name)
  string_list = cfg_file.readlines()
  cfg_file.close()
  print(string_list)
  print(string_list[0])
  #string_list[7]='width = 416\n'
  #string_list[8]='height = 416\n'
  string_list[19]='max_batches = 6000\n'
  string_list[5] = 'batch = 24\n'
  string_list[6] = 'subdivisions = 16\n'
  string_list[21]='steps = 4800,5400\n'
  string_list[1057]='classes=1\n'
  string_list[1050]='filters = 18\n'
  string_list[969]='classes=1\n'
  string_list[962]='filters = 18\n'
  string_list[1145]='classes=1\n'
  string_list[1138]='filters = 18\n'

  cfg_file = open(comp_name,'w')
  new_file_contents = ''.join(string_list)
  cfg_file.write(new_file_contents)
  cfg_file.close()
  readable_file = open(comp_name)
  read_file = readable_file.read()
  print(read_file) 
  return

     
def create_file_name(docker):
  execute_in_docker = True    
  save_path = 'darknet/data' if execute_in_docker else './darknet/data'
  file_name = 'obj.names'
  comp_name = os.path.join(save_path,file_name)
  with open(comp_name,'w') as f:
    f.write('nodules')
  return 


def create_file_data(docker):
    execute_in_docker = True  
    import os
    save_path = 'darknet/data' if execute_in_docker else './darknet/data'
    file_name = 'obj.data'
    comp_name = os.path.join(save_path,file_name)

    with open(comp_name,'w') as f:
        f.write('classes = 1\n')
        f.write('train = '+ save_path +'/train.txt\n')
        f.write('valid = '+ save_path + '/test.txt\n')
        f.write('names = '+save_path +'/obj.names\n')
        name = os.path.join('/output','model_retrained')
        if os.path.isdir(name) == False:
            os.mkdir(name)
        f.write('backup = '+ name)