import SimpleITK
import numpy as np

from pandas import DataFrame
from scipy.ndimage import center_of_mass, label
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from evalutils import DetectionAlgorithm
from evalutils.validators import (
    UniquePathIndicesValidator,
    UniqueImagesValidator,
)
from skimage import transform
import json
from typing import Dict
#import training_utils.utils as utils
#from training_utils.dataset import CXRNoduleDataset, get_transform
import os
#from training_utils.train import train_one_epoch
import itertools
from pathlib import Path
from postprocessing import get_NonMaxSup_boxes
from preprocessing import preprocess_data_images
from preprocessing import preprocess
from preprocessing import splitting
from create_dirs import create_darknet_dirs
from create_files import create_file_cfg, create_file_name, create_file_data
from getting_results import get_result
# This parameter adapts the paths between local execution and execution in docker. You can use this flag to switch between these two modes.
# For building your docker, set this parameter to True. If False, it will run process.py locally for test purposes.
execute_in_docker = False

class Noduledetection(DetectionAlgorithm):
    def __init__(self, input_dir, output_dir, train=False, retrain=False, retest=False):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
            input_path = Path(input_dir),
            output_file = Path(os.path.join(output_dir,'nodules.json'))
            #output_path = Path(output_dir)
        )
        
        #------------------------------- LOAD the model here ---------------------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('using the device ', self.device)
        #self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s',autoshape = False)
        num_classes = 1  # 1 class (nodule) + background
        #in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        #self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        path1 = 'opt/algorithm/darknet/cfg/yolov4-obj.cfg' if execute_in_docker else './darknet/cfg/yolov4-obj.cfg'
        if os.path.isfile(path1) ==False:
            create_file_cfg(execute_in_docker)

        path2 = 'opt/algorithm/darknet/data/obj.names' if execute_in_docker else './darknet/data/obj.names'
        if os.path.isfile(path2)== False:
            create_file_name(execute_in_docker) 
        
        path3 = 'opt/algorithm/darknet/data/obj.data' if execute_in_docker else './darknet/data/obj.data'
        if os.path.isfile(path3) == False:
            create_file_data(execute_in_docker)

        #wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
        
        if not (train or retest):
            # retrain or test phase
            print('loading the model from container with model file:')
            model_weights = os.path.join('opt/algortihm','model','yolov4-obj_last.weights') if execute_in_docker else './model/yolov4-obj_last.weights'   
            
        if retest:
            print('loading the retrained model for retest phase')
            
            model_weights = os.path.join('/output','model_retrained','yolov4-obj_last.weights')
    
    
        #self.model.to(self.device)
        
    def save(self):
        with open(str(self._output_file), "w") as f:
            json.dump(self._case_results[0], f)

      # TODO: Copy this function for your processor as well!
    def process_case(self, *, idx, case):
        '''
        Read the input, perform model prediction and return the results. 
        The returned value will be saved as nodules.json by evalutils.
        process_case method of evalutils
        (https://github.com/comic/evalutils/blob/fd791e0f1715d78b3766ac613371c447607e411d/evalutils/evalutils.py#L225) 
        is overwritten here, so that it directly returns the predictions without changing the format.
        
        '''
        # Load and test the image for this case
        input_image, input_image_file_path = self._load_input_image(case=case)
        
        # Detect and score candidates
        scored_candidates = self.predict(input_image=input_image)
        
        # Write resulting candidates to nodules.json for this case
        return scored_candidates

        
    #--------------------Write your retrain function here ------------
    def train(self,input_dir,output_dir,num_epochs = 1):
        '''
        input_dir: Input directory containing all the images to train with
        output_dir: output_dir to write model to.
        num_epochs: Number of epochs for training the algorithm.
        '''
        # Implementation of the pytorch model and training functions is based on pytorch tutorial: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html

        # create training dataset and defined transformations
        #input_dir = self.input_path
        #output_dir = self.output_path
        print('training starts ')
        create_darknet_dirs()
        preprocess_data_images(input_dir,os.path.join(input_dir,'metadata.csv'))
        print('splitting data')
        splitting()
        #write training commands here
        # save retrained version frequently.
        #print('saving the model')
        #torch.save(self.model.state_dict(), Path("/output/model_retrained.pth") if execute_in_docker else os.path.join(output_dir, 'm    
        os.system('darknet detector train darknet/data/obj.data darknet/cfg/yolov4-obj.cfg yolov4.conv.137 -dont_show -map')
    
    def predict(self, *, input_image: SimpleITK.Image) -> DataFrame:
        image_data = SimpleITK.GetArrayFromImage(input_image)
        spacing = input_image.GetSpacing()
        print('spacing',spacing)
        print(type(spacing))
        image_data = np.array(image_data)
        save_path2 = '/opt/algorithm/test1' if execute_in_docker else './test1'
        if os.path.isdir(save_path2)== False:
            os.mkdir(save_path2)
        if len(image_data.shape)==2:
            
            image_data = np.expand_dims(image_data, 0)
            print('after expanding',image_data.shape)
            print('len',len(image_data))
        
        name2 = '/opt/algorithm/test1.txt' if execute_in_docker else './test1.txt'
        with open(name2,'a') as f:
           # operate on 3D image (CXRs are stacked together)
           for j in range(len(image_data)):
            # Pre-process the image
            image = image_data[j,:,:]
            print(image.shape)
            preprocess(image,j)
            name1 = '/opt/algorithm/test1' if execute_in_docker else './test1'
            f.write(name1+str(j)+'.png\n')
            # write testing command here   
            os.system('darknet detector test darknet/data/obj.data darknet/cfg/yolov4-obj.cfg model_weights -ext_output -dont_show -out result.json <test1.txt -thresh 0.04')   
        data2 = get_result(spacing)
        print(data2)
        return(data2) 

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        prog='process.py',
        description=
            'Reads all images from an input directory and produces '
            'results in an output directory')

    parser.add_argument('input_dir', help = "input directory to process")
    parser.add_argument('output_dir', help = "output directory generate result files in")
    parser.add_argument('--train', action='store_true', help = "Algorithm on train mode.")
    parser.add_argument('--retrain', action='store_true', help = "Algorithm on retrain mode (loading previous weights).")
    parser.add_argument('--retest', action='store_true', help = "Algorithm on evaluate mode after retraining.")

    parsed_args = parser.parse_args()  
    if (parsed_args.train or parsed_args.retrain):# train mode: retrain or train
        Noduledetection(parsed_args.input_dir, parsed_args.output_dir, parsed_args.train, parsed_args.retrain, parsed_args.retest).train(parsed_args.input_dir,parsed_args.output_dir)
    else:# test mode (test or retest)
        Noduledetection(parsed_args.input_dir, parsed_args.output_dir, retest=parsed_args.retest).process()
            
    
   
    
    
    
    
    
