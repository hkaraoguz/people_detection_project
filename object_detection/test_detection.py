import os
import time
import sys
import torchvision
import numpy as np
import torch
import datetime


sys.path.append('/home/hakan/development/git_repos/people_detection_project/object_detection/sort')
print(sys.path)
from sort import *
#import torch
#import numpy as np
#from torchvision import transforms

from cv2 import cv2

class Person(object):

    def __init__(self, tracking_id):
        self.frames  = []
        self.tracking_id = tracking_id
        self.boxes = []
        self.first_timestamp = None



# torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
# torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
#model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True, min_size=400, max_size=600)
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)


dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#dev = torch.device("cpu")
print(dev)
model.to(dev)
model.eval()



COCO_INSTANCE_CATEGORY_NAMES = ['__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

print(len(COCO_INSTANCE_CATEGORY_NAMES))

url = 'http://91.151.111.152:8080/axis-cgi/mjpg/video.cgi?camera=&resolution=640x480'

#'http://188.9.157.80:81/axis-cgi/mjpg/video.cgi?camera=&resolution=640x480'#'http://73.13.148.126:8082/mjpg/video.mjpg'#'http://72.43.190.171:81/mjpg/video.mjpg'

#'http://184.153.62.129:82/mjpg/video.mjpg' #'http://189.174.160.218:6001/mjpg/video.mjpg'
#'http://180.44.188.214:3000/mjpg/video.mjpg'#'http://67.250.253.16:8081/mjpg/video.mjpg' #'http://82.58.15.68:83/mjpg/video.mjpg' #http://122.58.10.67:83/mjpg/video.mjpg


# Transform the image to tensor
tran = torchvision.transforms.ToTensor()
color = (255, 0, 0)
thickness = 2

frames = []
start_time = None
end_time = None
last_time = None
person_counter = 0
frame_counter = 0
person_boxes = []

MAX_TIME = 10

curr_path = os.path.dirname(os.path.abspath(__file__))

pictures_path = os.path.join(curr_path,'pictures')

if not os.path.exists(pictures_path):
    os.mkdir(pictures_path)
def visualize_detections(input_frame, dets):

    for box in dets:
        start_point = (int(box[0]), int(box[1]))
        end_point = (int(box[2]), int(box[3]))
        input_frame = cv2.rectangle(input_frame, start_point, end_point, color, thickness)

    return input_frame


def get_people_tracks(tracker, dets):

    trackers = tracker.update(dets)

    return trackers
   


def get_detections(frame, model):

    process_frame = frame

    process_frame = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB )


    # Process the image
    tens = tran(process_frame)

    tens = tens.to(dev)

    try:
        predictions = model([tens])

        #predictions = predictions.cpu()

        if predictions:

            boxes = predictions[0]['boxes'].cpu().detach().numpy()
            scores = predictions[0]['scores'].cpu().detach().numpy()
            label_indexes = predictions[0]['labels'].cpu().detach().numpy()

            return boxes, scores, label_indexes

        return np.zeros(1), None, None

    except Exception as ex:
        print(ex)
        return np.zeros(1), None, None

def track_people(trackers, frame):
    ids = []
    print('trackers ',trackers)
    #print(type(trackers))
    if 'numpy' in str(type(trackers)):
        #print(trackers)
        if trackers.size > 0:
            ids = trackers[:,4]
            ids = ids.tolist()
    else:
        return
            

    # get unique ids
    ids = list(set(ids))

    print(ids)

    for person_id in ids:
        
        float_id = person_id
        person_id = int(person_id)
        
        if not people_track_start_dict.get(person_id):
            
            indices = np.where(trackers[:,4] == float_id)[0]
            person = Person(person_id)
            person_box = None
            if indices.size >0:
                person_box = trackers[indices[-1],:4].astype(int).tolist()
                person.boxes.append(person_box)
                #print(person_box)
            if person_box:
                new_frame = visualize_detections(frame.copy(),[person_box])
                people_track_start_dict[person_id] = time.time()
                person.frames.append(new_frame)
                person.first_timestamp = time.time()
                people_dict[person_id] = person
        else:
            person = people_dict[person_id]
            #person.frames.append(frame)
            indices = np.where(trackers[:,4] == float_id)[0]
            person_box = None

            if indices.size >0:
                person_box = trackers[indices[-1],:4].astype(int).tolist()
                person.boxes.append(person_box)
                new_frame = visualize_detections(frame.copy(),[person_box])
                person.frames.append(new_frame)
            
            people_dict[person_id] = person

    del_indexes = []
    for key, value in people_track_start_dict.items():
        if time.time() - value >= MAX_TIME:
                person = people_dict[key]
                print(person)

                if len(person.frames) >= 3:
                    print(person.first_timestamp)
                    print(person.tracking_id)

                del_indexes.append(key)

    for index in del_indexes:
        del people_dict[index]
        del people_track_start_dict[index]



#create instance of the SORT tracker
mot_tracker = Sort(max_age=10,
                       min_hits=2,iou_threshold=0.2)

people_track_start_dict = dict()

people_dict = dict()


stream = cv2.VideoCapture(url)

trackers = []

# infinite loop
while True:

    # read frames
    (grabbed, frame) = stream.read()


    # check if frame empty
    if not grabbed:
        print('opencv is buggy')
        continue
        #break
    
    boxes, scores, label_indexes = get_detections(frame,model)

    person_boxes = []
    
    #print(predictions[0]['boxes'].detach().numpy())

    for index, box in enumerate(boxes):

        score = scores[index]
        label_index = label_indexes[index]
        label = COCO_INSTANCE_CATEGORY_NAMES[label_index]

        if score > 0.7 and label=='person':

            person_counter += 1

            new_box = box.astype(int)

            new_box = new_box.tolist()

            new_box.append(score)

            person_boxes.append(new_box)

            #print('score: ',score,' label: ',label)
            #break
    if person_boxes:
        trackers = get_people_tracks(mot_tracker, np.asarray(person_boxes))
        #track_people(trackers,frame.copy())
    else:
        trackers =get_people_tracks(mot_tracker, np.empty((0,5)))
 
    track_people(trackers,frame.copy())
    frame = visualize_detections(frame,person_boxes)

    cv2.imshow("Output Frame", frame)
    
    key = cv2.waitKey(1) & 0xFF
    # check for 'q' key-press
    if key == ord("q"):
        #if 'q' key-pressed break out
        break
    
    

cv2.destroyAllWindows()
# close output window