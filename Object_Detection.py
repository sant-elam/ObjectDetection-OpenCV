import cv2 as cv
import numpy as np


def post_process(image,
                 outputs,
                 target_width,
                 target_height,
                 conf_threshold,
                 nms_threshold):
    boxes  = []
    probabilities = []
    class_id =[]

    nms_indices = []

    height = image.shape[0]
    width = image.shape[1]

    #height_factor = height / target_height
    # width_factor = width / target_width

    prob_index = 5

    # Scan through all the boxes and keep the one with high confidence score greater than confidence_thres...
    for output in outputs:
        prob_array = output[prob_index:]

        conf_index = np.argmax(prob_array, axis=-1)
        conf_score = prob_array[conf_index]

        if conf_score > conf_threshold and conf_index == 0:
            print("-------------------------------")
            print(len(prob_array))
            print(len(output))
            print("-------------------------------")
            probabilities.append(float(conf_score))
            class_id.append(conf_index)

            # get the box values, x, y, width, height to left, top, width, height
            # get the values and convert to actual co-ordinates
            x_center = output[0] * width
            y_center = output[1] * height
            width_box = output[2] * width
            height_box = output[3] * height

            # convert to left, top, width and height
            left = int(x_center - (width_box * 0.5))
            top = int(y_center - (height_box * 0.5))
            width_box = int(width_box)
            height_box = int(height_box)


            boxes.append([left, top, width_box, height_box])

    # NMS Filtering
    no_of_boxes = len(boxes)
    if no_of_boxes > 0:
        nms_indices = cv.dnn.NMSBoxes(boxes,
                                      probabilities,
                                      conf_threshold,
                                      nms_threshold)

    return boxes, nms_indices, class_id, probabilities


def display_box(image,
                boxes,
                indices,
                class_ids,
                scores,
                class_names,
                colors):

    no_of_boxes = len(boxes)

    if(no_of_boxes>0):

        for i in indices:
            index = int(i)

            class_id = class_ids[index]
            class_name = class_names[class_id]
            score = scores[index]

            box = boxes[index]
            left, top, width, height = box[:4]

            color_index = class_id % len(colors)
            color = colors[int(color_index)]

            text = "{}  {:.2f}".format(class_name, score)

            top_display = top
            if top_display < 20:
                top_display = top + height + 10
            else:
                top_display = top_display - 10

            cv.rectangle(image,
                         (left, top),
                         (left+width, top+height),
                         color,
                         1)
            cv.putText(image,
                       text,
                       (left, top_display),
                       cv.FONT_HERSHEY_SIMPLEX,
                       0.5,
                       color,
                       1)



RESIZE_WIDTH = 416
RESIZE_HEIGHT = 416
CONF_THRES = 0.6
NMS_THRES = 0.5

COLORS    = [(50, 200, 100),
             (100, 255, 100),
             (100, 10, 255),
             (255, 10, 100),
             (100, 255, 255)]


# 1.....LOAING THE VIDEO.........
video_path = 'video2.mp4'
capture = cv.VideoCapture(video_path)



# 2.....LOADING THE MODEL .............
# a........ Loading the configuration file (yolo.cfg)
# b........ Loading the model (yolo.weights)

config_file = 'yolov3.cfg'
model = 'yolov3.weights'

read_net = cv.dnn.readNetFromDarknet(config_file, model)

# 3......SELECT SPECIFIC COMPUTATION........
read_net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

# 4......LOAD THE CLASS NAMES......
class_file = 'coco.names'

# read all lines from the class file
all_lines = open(class_file).read()
# remove spaces
all_lines = all_lines.strip()
# convert strings into list
classes = all_lines.split('\n')
# displaying all classes
print(len(classes),classes)

# 5......GETTING THE NETWORK OUTPUT LAYERS.......
#   a...... Get the Layers .....
layers = read_net.getLayerNames()
print(len(layers),layers)

#   b...... Get the Output layers ....
#           There are 3 output layers
unconnected_layer=[]
for i in read_net.getUnconnectedOutLayers():
    unconnected_layer.append(layers[int(i-1)])

print(unconnected_layer)

# 6.....GETTING THE IMAGES

index = 0
while True:
    #  a... Read image from capture
    has_frame, image = capture.read()

    if has_frame and index % 3:
        #  b... Convert image to RGB color format
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        #  c... Construct a blob image...
        blob_image = cv.dnn.blobFromImage(image_rgb,
                                          1.0 / 255.0,
                                          (RESIZE_WIDTH, RESIZE_HEIGHT),
                                          swapRB=True,
                                          crop=False)


        #  d... Pass the blob image as inputs to the network
        read_net.setInput(blob_image)

        #  e... Get the list of predicted output box

        #       Output returns a vector array containing
        #       5 elements + number of class

        #        0          1       2       3        4         5        6        7        8           N
        #     center_x, center_y, width, height, confidence, class_1, class_2, class_3, class_4,.... class_n

        #     center_x      : x cordinate of the center of the bounding box
        #     center_y      : y cordinate of the center of the bounding box
        #     width         : width of the bounding box
        #     height        : height of the bounding box
        #     confidence    : The confidence of the object enclosed by the bounding box
        #     the rest      : The rest of the elements are the confidence of associated with each of the classes

        #     The box is assigned the class having the highest score. It is also called confidence of the box.

        outputs = read_net.forward(unconnected_layer)

        #print('1:', outputs[0])
        #print('2:', outputs[1])
        #print('3:', outputs[2])
        #  f... Combine the 3 outputs into 1

        #  YOLO OUTPUTS has 3 outputs for handling different sizes

        #   i. 507 ( 13 x 13 x 3)  : For handling large objects
        #  ii. 2028 ( 26 x 26 x 3) : For medium objects
        # iii. 8112 ( 52 x 52 x 3) : For small objects

        outputs_combined = np.vstack(outputs)

        #  g... Post-Processing...
        #       i. In first steps, it filters the boxes with low confidence score less than a given threshold
        #      ii. In the second process, it reduces the overlapping boxes by applying Non Maxima Supression.

        boxes, nms_index, class_ids, probabilities = post_process(image,
                                                                  outputs_combined,
                                                                  RESIZE_WIDTH,
                                                                  RESIZE_HEIGHT,
                                                                  CONF_THRES,
                                                                  NMS_THRES)

        # h.. Display Class and Score

        display_box(image,
                    boxes,
                    nms_index,
                    class_ids,
                    probabilities,
                    classes,
                    COLORS)

        cv.imshow("Object Detection", image)
        cv.waitKey(1)

    index = index + 1
cv.destroyAllWindows()
capture.release()