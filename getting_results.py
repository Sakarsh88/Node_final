from typing import Dict
def get_result(spacing2):
    import json
    name = '/opt/algortihm/darknet/result.json' if execute_in_docker else './darket/result.json'
    f = open(name)
    data = json.load(f)
    merged_d = {}
    data1 = []
    listobj = []
    np_prediction = {}
    spacing = spacing2
    print(type(spacing))
    for frame_no,i in enumerate(data):
        print('frame_no',frame_no)#this will basically be the slice no
        bjects = i['objects']
        filename = i['filename']
        boxes = []
        scores = []
        results = []
        for detection_no,detections in enumerate(objects):
            relative_coordinates = detections['relative_coordinates']
            score = detections['confidence']
            centre_x = relative_coordinates['center_x']
            centre_y = relative_coordinates['center_y']
            width = relative_coordinates['width']
            height = relative_coordinates['height']
            x1 = (centre_x - width/2)*608
            y1 = (centre_y - height/2)*608
            x2 = (centre_x + width/2)*608
            y2 = (centre_y + height/2)*608
            boxes.append([x1,y1,x2,y2])
            scores.append(score)
            np_prediction ={'boxes':boxes,'scores':scores}
            prediction = get_NonMaxSup_boxes(np_prediction)
        if np_prediction:
            pass
        else:
            continue    
        np_prediction['slice']=len(np_prediction['boxes'])*[frame_no]
        np_prediction_df = pd.DataFrame(np_prediction)
        results.append(np_prediction)
  
        for k in results[0].keys():
            merged_d[k] = list(itertools.chain(*[d[k] for d in results]))
        predictions = merged_d

        data = format_to_GC(predictions,spacing)
        data1.append(data)
        if comp_name.is_file() is False:
            with open(comp_name,'w') as f2:
                json.dumps(data,f2,indent=2)    
        else:
            with open(comp_name,'r+') as file:
                listobj = list(json.load(file))
                listobj.append([data])
                print(listobj)
                file.seek(0)
                json.dump(listobj,file,indent=2)
    return data1   
    

    
 
def get_NonMaxSup_boxes(pred_dict):
        scores = pred_dict['scores']
        boxes = pred_dict['boxes']
        lambda_nms = 0.3

        out_scores = []
        out_boxes = []
        for ix, (score, box) in enumerate(zip(scores,boxes)):
            discard = False
            for other_box in out_boxes:
              
              if intersection_over_union(box, other_box) > lambda_nms:
                    discard = True
                    break
            if not discard:
                out_scores.append(score)
                out_boxes.append(box)
        return {'scores':out_scores, 'boxes':out_boxes}
    
# Source: https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    #print(interArea)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou  

def format_to_GC(np_prediction, spacing) -> Dict:
  '''
  Convenient function returns detection prediction in required grand-challenge format.
        See:
        https://comic.github.io/grandchallenge.org/components.html#grandchallenge.components.models.InterfaceKind.interface_type_annotation
        
        
        np_prediction: dictionary with keys boxes and scores.
        np_prediction[boxes] holds coordinates in the format as x1,y1,x2,y2
        spacing :  pixel spacing for x and y coordinates.
        
        return:
        a Dict in line with grand-challenge.org format.
        '''
        # For the test set, we expect the coordinates in millimeters. 
        # this transformation ensures that the pixel coordinates are transformed to mm.
        # and boxes coordinates saved according to grand challenge ordering.
  x_y_spacing = [spacing[0], spacing[1], spacing[0], spacing[1]]
  boxes = []
  for i, bb1 in enumerate(np_prediction['boxes']):
    box = {}   
    box['corners']=[]
    bb = np.asarray(bb1)
    #print(bb)
    #print(type(bb))
    #print(np_prediction['slice'][i])
    #print(i)
    x_min, y_min, x_max, y_max = bb*x_y_spacing
    x_min, y_min, x_max, y_max  = round(x_min, 2), round(y_min, 2), round(x_max, 2), round(y_max, 2)
    bottom_left = [x_min, y_min,  np_prediction['slice'][i]] 
    bottom_right = [x_max, y_min,  np_prediction['slice'][i]]
    top_left = [x_min, y_max,  np_prediction['slice'][i]]
    top_right = [x_max, y_max,  np_prediction['slice'][i]]
    box['corners'].extend([top_right, top_left, bottom_left, bottom_right])
    box['probability'] = round(float(np_prediction['scores'][i]), 2)
    boxes.append(box)
        
  return dict(type="Multiple 2D bounding boxes", boxes=boxes, version={ "major": 1, "minor": 0 })
        