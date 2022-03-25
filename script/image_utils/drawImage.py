import cv2
import numpy as np

def draw_grid(img, grid_shape, color=(0, 255, 0), thickness=1):
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols

    # draw vertical lines
    for x in np.linspace(start=dx, stop=w-dx, num=cols-1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)

    # draw horizontal lines
    for y in np.linspace(start=dy, stop=h-dy, num=rows-1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)

    return img

def gen_boxes(img,num_boxes=1,box_size=(100,300),box_shape=(2,3),color=(0,255,0),thickness=1):
    h, w, _ = img.shape
    #print(f"h:{h},w:{w}")
    origin = (350,25)#(w//3,h//3)
    boxes = []
    for i in range(box_shape[0]):
        for j in range(box_shape[1]):
            x0 = origin[0] + j*box_size[0]
            y0 = origin[1] + i*box_size[1]
            x1 = x0 + box_size[0]
            y1 = y0 + box_size[1]
            boxes.append((x0,y0,x1,y1))
    return boxes

def draw_boxes(img,boxes,color=(0,255,0),thickness=1):
    id = 1
    for x0,y0,x1,y1 in boxes:
        cv2.rectangle(img, (x0,y0), (x1,y1), color=color, thickness=thickness)
        #put id in each box
        cv2.putText(img,str(id),(x0,y0),cv2.FONT_HERSHEY_SIMPLEX,1,color=(255,0,0),thickness=3)
        id += 1
    return img
   


def draw_boxes_at_center(img,num_boxes=1,box_size=100,color=(0,255,0),thickness=1):
    h, w, _ = img.shape
    dy, dx = h / num_boxes, w / num_boxes
    for i in range(num_boxes):
        x, y = int(round(i*dx)), int(round(i*dy))
        cv2.rectangle(img, (x, y), (x+box_size, y+box_size), color=color, thickness=thickness)
    return img