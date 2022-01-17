import torch
import glob

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


model = torch.hub.load('ultralytics/yolov5', 'custom', path='modelSmall/exp/weights/best.pt')

#img = "test_im/0001.png"
img = sorted(glob.glob("test_im/*", recursive=True))
ann = sorted(glob.glob("test_ann_s/*", recursive=True))

iou_sum = 0
l = 0

out = []
for id, i in enumerate(img):
    print(f"{id}/{len(img)}\r")
    values = open(ann[id], "r").readline().strip().split(" ")
    a = {"x1": int(values[1]), "y1": int(values[2]),"x2": int(values[1]) + int(values[3]), "y2": int(values[2]) + int(values[4])}
    res = model(i)

    #print(res.xyxy)
    out.append(res.xywh)

    try: 
        r = res.xyxy[0].tolist()[0]
        b = {"x1": int(r[0]), "y1": int(r[1]),"x2": int(r[2]), "y2": int(r[3])}
        iou_sum += get_iou(a, b)
    except IndexError:
        iou_sum += 0
    
    #print(iou_sum)
    l += 1
    #print(values)
    #print(r)

print(iou_sum/l)

#large -> 0.844 IOU
#mediu -> 0.835 IOU
#small -> 0.807 IOU