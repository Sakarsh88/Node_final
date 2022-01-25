import os
def create_darknet_dirs():
    execute_in_docker = True
    save_path = 'darknet/data/' if execute_in_docker else './darknet/data'
    file_name1 = 'images_nodules'
    comp_name = os.path.sep.join([save_path,file_name1])
    if os.path.isdir(comp_name)==False :
        os.mkdir(comp_name)

    file_name2 = 'images_no_nodules'
    comp_name2 = os.path.sep.join([save_path,file_name2])
    if os.path.isdir(comp_name2)==False :
        os.mkdir(comp_name2)

    file_name3 = 'annotations'
    comp_name3 = os.path.sep.join([save_path,file_name3])
    if os.path.isdir(comp_name3)==False :
        os.mkdir(comp_name3)

    file_name4 = 'obj'
    comp_name4 = os.path.sep.join([save_path,file_name4])
    if os.path.isdir(comp_name4)==False :
        os.mkdir(comp_name4)
    
    return