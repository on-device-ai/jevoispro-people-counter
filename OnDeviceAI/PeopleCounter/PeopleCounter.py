import pyjevois
if pyjevois.pro: import libjevoispro as jevois
else: import libjevois as jevois
import cv2 as cv
import numpy as np
from PIL import Image
from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter
import time

import dlib

from scipy.spatial import distance as dist
from collections import OrderedDict

class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

        # store the maximum distance between centroids to associate
        # an object -- if the distance is larger than this maximum
        # distance we'll start to mark the object as "disappeared"
        self.maxDistance = maxDistance

    def register(self, centroid):
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                # if we have reached a maximum number of consecutive
                # frames where a given object has been marked as
                # missing, deregister it
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])

        # otherwise, are are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
            # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue

                # if the distance between centroids is greater than
                # the maximum distance, do not associate the two
                # centroids to the same object
                if D[row, col] > self.maxDistance:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col])

        # return the set of trackable objects
        return self.objects

class TrackableObject:
    def __init__(self, objectID, centroid):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]

        # initialize a boolean used to indicate if the object has
        # already been counted or not
        self.counted = False

## 
#
# This module is here for you to experiment with Python OpenCV on JeVois and JeVois-Pro.
#
# By default, we get the next video frame from the camera as an OpenCV BGR (color) image named 'inimg'.
# We then apply some image processing to it to create an overlay in Pro/GUI mode, an output BGR image named
# 'outimg' in Legacy mode, or no image in Headless mode.
#
# - In Legacy mode (JeVois-A33 or JeVois-Pro acts as a webcam connected to a host): process() is called on every
#   frame. A video frame from the camera sensor is given in 'inframe' and the process() function create an output frame
#   that is sent over USB to the host computer (JeVois-A33) or displayed (JeVois-Pro).
#   
# - In Pro/GUI mode (JeVois-Pro is connected to an HDMI display): processGUI() is called on every frame. A video frame
#   from the camera is given, as well as a GUI helper that can be used to create overlay drawings.
#
# - In Headless mode (JeVois-A33 or JeVois-Pro only produces text messages over serial port, no video output):
#   processNoUSB() is called on every frame. A video frame from the camera is given, and the module sends messages over
#   serial to report what it sees.
#
# Which mode is activated depends on which VideoMapping was selected by the user. The VideoMapping specifies camera
# format and framerate, and what kind of mode and output format to use.
#
# See http://jevois.org/tutorials for tutorials on getting started with programming JeVois in Python without having
# to install any development software on your host computer.
#
# @author Yi-Lin Tung
# 
# @videomapping JVUI 0 0 0 CropScale=RGB3@512x288:YUYV 1920 1080 30 OnDeviceAI PeopleCounter
# @email yilintung@on-device-ai.com
# @address fixme
# @copyright Copyright (C) 2021 by Yi-Lin Tung
# @mainurl https://on-device-ai.com
# @supporturl 
# @otherurl 
# @license 
# @distribution Unrestricted
# @restrictions None
# @ingroup modules
class PeopleCounter:
    # ###################################################################################################
    ## Constructor
    def __init__(self):
        self.threshold = 0.6 # Confidence threshold (0..1), higher for stricter confidence.
        self.rgb = True      # True if model expects RGB inputs, otherwise it expects BGR
        
        # Select one of the models:
        self.model = 'MobileDetSSD' # expects 320x320

        # You should not have to edit anything beyond this point.
        if (self.model == 'MobileDetSSD'):
            classnames = 'coco_labels.txt'
            modelname = 'ssdlite_mobiledet_coco_qat_postprocess_edgetpu.tflite'

        # Load names of classes:
        sdir = pyjevois.share + '/coral/detection/'
        self.labels = read_label_file(sdir + classnames)

        # Load network:
        self.interpreter = make_interpreter(sdir + modelname)
        #self.interpreter = make_interpreter(*modelname.split('@'))
        self.interpreter.allocate_tensors()
        self.timer = jevois.Timer('Coral classification', 10, jevois.LOG_DEBUG)
        
        
        # initialize the frame dimensions (we'll set them as soon as we read
        # the first frame from the video)
        self.W = None
        self.H = None

        # skip frames between detections
        self.skip_frames = 15

        # instantiate our centroid tracker, then initialize a list to store
        # each of our dlib correlation trackers, followed by a dictionary to
        # map each unique object ID to a TrackableObject
        self.ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        self.trackers = []
        self.trackableObjects = {}

        # initialize the total number of frames processed thus far, along
        # with the total number of objects that have moved either up or down
        self.totalFrames = 0
        self.totalDown = 0
        self.totalUp = 0
             
        # ####################################################################################################
    ## Process function with GUI output
    def processGUI(self, inframe, helper):
        # Start a new display frame, gets its size and also whether mouse/keyboard are idle:
        idle, winw, winh = helper.startFrame()

        # Draw full-resolution input frame from camera:
        x, y, w, h = helper.drawInputFrame("c", inframe, False, False)
        
        # Get the next camera image at processing resolution (may block until it is captured):
        frame = inframe.getCvRGBp() if self.rgb else inframe.getCvBGRp()

        # Start measuring image processing time:
        self.timer.start()

        # Set the input:
        image = Image.fromarray(frame);
        _, scale = common.set_resized_input(self.interpreter, image.size,
                                            lambda size: image.resize(size, Image.ANTIALIAS))
                                            
        # if the frame dimensions are empty, set them
        if self.W is None or self.H is None:
            self.W, self.H = image.size
            
        # initialize the current status along with our list of bounding
        # box rectangles returned by either (1) our object detector or
        # (2) the correlation trackers
        status = "Waiting"
        rects = []
                
        # check to see if we should run a more computationally expensive
        # object detection method to aid our tracker
        if self.totalFrames % self.skip_frames == 0:
            # set the status and initialize our new set of object trackers
            status = "Detecting"
            self.trackers = []
            
            # Run the model
            start = time.perf_counter()
            self.interpreter.invoke()
            inference_time = time.perf_counter() - start
            
            # Get detections with high enough scores:
            objs = detect.get_objects(self.interpreter, self.threshold, scale)
            
            for obj in objs:
                bbox = obj.bbox
                label = self.labels.get(obj.id, obj.id)
                
                # if the class label is not a person, ignore it
                if label != "person":
                    continue
                
                # compute the (x, y)-coordinates of the bounding box
                # for the object
                box = np.array([bbox.xmin,bbox.ymin,bbox.xmax,bbox.ymax])
                (startX, startY, endX, endY) = box.astype("int")
                
                # construct a dlib rectangle object from the bounding
                # box coordinates and then start the dlib correlation
                # tracker
                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(startX, startY, endX, endY)
                tracker.start_track(np.array(image), rect)

                # add the tracker to our list of trackers so we can
                # utilize it during skip frames
                self.trackers.append(tracker)
        
        # otherwise, we should utilize our object *trackers* rather than
        # object *detectors* to obtain a higher frame processing throughput
        else:
            # loop over the trackers
            for tracker in self.trackers:
                # set the status of our system to be 'tracking' rather
                # than 'waiting' or 'detecting'
                status = "Tracking"

                # update the tracker and grab the updated position
                tracker.update(np.array(image))
                pos = tracker.get_position()

                # unpack the position object
                startX = int(pos.left())
                startY = int(pos.top())
                endX = int(pos.right())
                endY = int(pos.bottom())

                # add the bounding box coordinates to the rectangles list
                rects.append((startX, startY, endX, endY))

        # draw a horizontal line in the center of the frame -- once an
        # object crosses this line we will determine whether they were
        # moving 'up' or 'down'
        line_len = self.W
        line_pos = self.H // 2
        helper.drawLine(float(0),float(line_pos-1),float(line_len),float(line_pos-1),0xff00ffff)
        helper.drawLine(float(0),float(line_pos),float(line_len),float(line_pos),0xff00ffff)
        helper.drawLine(float(0),float(line_pos+1),float(line_len),float(line_pos+1),0xff00ffff)
        
        # use the centroid tracker to associate the (1) old object
        # centroids with (2) the newly computed object centroids
        objects = self.ct.update(rects)
        
        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = self.trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                direction = centroid[1] - np.mean(y)
                to.centroids.append(centroid)

                # check to see if the object has been counted or not
                if not to.counted:
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if direction < 0 and centroid[1] < self.H // 2:
                        self.totalUp += 1
                        to.counted = True

                    # if the direction is positive (indicating the object
                    # is moving down) AND the centroid is below the
                    # center line, count the object
                    elif direction > 0 and centroid[1] > self.H // 2:
                        self.totalDown += 1
                        to.counted = True
            
            # store the trackable object in our dictionary
            self.trackableObjects[objectID] = to
            
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            text = "ID {}".format(objectID)
            helper.drawText(float(centroid[0] - 10), float(centroid[1] - 10), text, 0xff00ff00)
            helper.drawCircle(float(centroid[0]), float(centroid[1]), 4, 0xff00ff00, True)
            
        # construct a tuple of information we will be displaying on the
        # frame
        info = [
            ("Up", self.totalUp),
            ("Down", self.totalDown),
            ("Status", status),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            helper.drawText(float(10), float(self.H - ((i * 20) + 20)), text, 0xff0000ff)
            
        self.totalFrames += 1
        
        # Write frames/s info from our timer:
        fps = self.timer.stop()
        helper.iinfo(inframe, fps, winw, winh);
        
        # End of frame:
        helper.endFrame()





