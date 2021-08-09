import os
import time
import sys
import torchvision
import numpy as np
from pymongo import MongoClient, database
import gridfs
import torch
import datetime


sys.path.append('./sort')
print(sys.path)
from sort import *
#import torch
#import numpy as np
#from torchvision import transforms

from cv2 import cv2


torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class Person(object):

    def __init__(self, tracking_id):
        self.frames  = []
        self.tracking_id = tracking_id
        self.boxes = []
        self.first_timestamp = None
        self.last_timestamp = None


def load_model():
    """
    Loads Yolo5 model from pytorch hub.
    :return: Trained Pytorch model.
    """
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    #self.model.to(self.device)
    return model.to(torch_device), model.names



model,model_labels = load_model()
print(torch_device)
model.eval()

print(model_labels)
print('number of model categories: ', len(model_labels))

camera_url = os.environ.get('CAMERA_URL')
mongo_host = os.environ.get('MONGO_HOST', 'mongo')
mongo_port = os.environ.get('MONGO_PORT', 27017)
mongo_port = int(mongo_port)
confidence_threshold = float(os.environ.get('CONFIDENCE_THRESHOLD',0.7))

print('camera url is: ', camera_url)
print('mongo client is: ', mongo_host)
print('mongo port is: ', mongo_port)
print('confidence threshold is: ', confidence_threshold)


url = camera_url #'http://5.2.202.59:8084/axis-cgi/mjpg/video.cgi?camera=&resolution=640x480'#'http://73.13.148.126:8082/mjpg/video.mjpg'#'http://72.43.190.171:81/mjpg/video.mjpg'

#'http://184.153.62.129:82/mjpg/video.mjpg' #'http://189.174.160.218:6001/mjpg/video.mjpg'
#'http://180.44.188.214:3000/mjpg/video.mjpg'#'http://67.250.253.16:8081/mjpg/video.mjpg' #'http://82.58.15.68:83/mjpg/video.mjpg' #http://122.58.10.67:83/mjpg/video.mjpg



color = (255, 0, 0)
thickness = 2

# Transform the image to tensor
tran = torchvision.transforms.ToTensor()

frames = []
start_time = None
end_time = None
last_time = None
person_counter = 0
frame_counter = 0
person_boxes = []

MAX_TIME = 10


def initialize_mongo_client():

    try:
        mongo_client =  MongoClient(mongo_host, mongo_port,connect=True)
        db = mongo_client.get_database('people_detection_project')
        fs = gridfs.GridFS(db)

    except Exception as ex:
        print(str(ex))

        return None

    return db,fs


    #print(db.list_collection_names())

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

    try:
        predictions = model([process_frame])

        #predictions = predictions.cpu()

        if predictions:
            
            #print(predictions.pred[0].cpu().numpy()[:,-2])

            scores = predictions.pred[0].cpu().numpy()[:,-2]
            
            label_indexes, boxes = predictions.xyxy[0][:, -1].cpu().numpy(), predictions.xyxy[0][:, :-1].cpu().numpy()

            label_indexes = label_indexes.astype(int)

            #boxes = predictions[0]['boxes'].cpu().detach().numpy()
            #scores = predictions[0]['scores'].cpu().detach().numpy()
            #label_indexes = predictions[0]['labels'].cpu().detach().numpy()

            return boxes, scores, label_indexes

        return np.zeros(1), None, None

    except Exception as ex:
        print(ex)
        return np.zeros(1), None, None

def track_people(trackers, frame):

    # First 4 is the bounding box, last 1 is the id of the person
    # trackers  [[     536.29      281.39      569.71       368.9           9]
    # [     579.42      253.32      693.02      478.13           5]]



    ids = []
    #print(type(trackers))
    if 'numpy' in str(type(trackers)):
        #print(trackers)
        if trackers.size > 0:
            ids = trackers[:,4]
            ids = ids.tolist()
            

        # get unique ids
        ids = list(set(ids))

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
                person.last_timestamp = person.first_timestamp
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
                person.last_timestamp = time.time()
            
            people_dict[person_id] = person

    del_indexes = []
    for key, person in people_dict.items():
         if time.time() - person.last_timestamp >= MAX_TIME:
                person = people_dict[key]

                if person.last_timestamp - person.first_timestamp >= 3.0:
                    print(person.first_timestamp)
                    print(person.tracking_id)

                    half_len_frames = len(person.frames)/2

                    # convert ndarray to string
                    imageString = person.frames[int(half_len_frames)].tobytes()

                    # store the image
                    imageID = grid_fs.put(imageString, encoding='utf-8')
                    

                    mongo_dict =  vars(person)

                    mongo_dict['image_id'] = imageID
                    mongo_dict['image_shape'] = person.frames[0].shape
                    
                    del mongo_dict['frames']
                    del mongo_dict['boxes']
                    
                    #mongo_dict['framepath'] = folder_path
                    mongo_db['person'].insert_one(mongo_dict)
                del_indexes.append(key)

    for index in del_indexes:
        del people_dict[index]
        del people_track_start_dict[index]



#create instance of the SORT tracker
mot_tracker = Sort(max_age=10,
                       min_hits=3,iou_threshold=0.3)

people_track_start_dict = dict()

people_dict = dict()

mongo_db,grid_fs  = initialize_mongo_client()


if mongo_db == None:
    print('Mongodb cannot be initialized')
    sys.exit(-1)

trackers = []
stream = cv2.VideoCapture(url)

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
    
    for index, box in enumerate(boxes):

        score = scores[index]
        label_index = label_indexes[index]
        label = model_labels[label_index]

        if score >= confidence_threshold and label=='person':

            person_counter += 1

            new_box = box.astype(int)

            new_box = new_box.tolist()

            new_box.append(score)

            person_boxes.append(new_box)

            #print('score: ',score,' label: ',label)
            #break
    if person_boxes:
        #frame = visualize_detections(frame,person_boxes)
        #print(person_boxes)
        trackers = get_people_tracks(mot_tracker, np.asarray(person_boxes))

    else:
        trackers =get_people_tracks(mot_tracker, np.empty((0,5)))
    
    track_people(trackers,frame.copy())
    #frame = visualize_detections(frame,person_boxes)
    # Show output window
    #cv2.imshow("Output Frame", frame)
    '''
    key = cv2.waitKey(1) & 0xFF
    # check for 'q' key-press
    if key == ord("q"):
        #if 'q' key-pressed break out
        break
    '''

#cv2.destroyAllWindows()
# close output window