import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_bboxes(boxes, box_format="mid_point", reverse=False):
    # Create a Rectangle patch
    if box_format=="mid_point":
        for _box in boxes:
            box = [_box[1], _box[0], _box[3], _box[2]] if reverse else _box
            pt0 = (int(box[1]-box[3]/2), int(box[0]-box[2]/2))
            rect = patches.Rectangle(
                pt0, box[3], box[2],
                linewidth=2, edgecolor='r', facecolor="none",
            )
            plt.gca().add_patch(rect)
    elif box_format=="corners":
        for _box in boxes:
            box = [_box[1], _box[0], _box[3], _box[2]] if reverse else _box
            pt0 = box[1], box[0]
            rect = patches.Rectangle(
                pt0, box[3]-box[1], box[2]-box[0],
                linewidth=2, edgecolor='r', facecolor="none",
            )
            plt.gca().add_patch(rect)
    return

def plot_bboxes_and_masks(box_mask_list, alpha=0.5):
    
    def get_color(value):
        v = np.clip(value, 0, 1)
        if v<0.2:
            return [1, v/0.2, 0]
        elif v<0.4:
            return [(0.4-v)/0.2, 1, 0]
        elif v<0.6:
            return [0, 1, (v-0.4)/0.2]
        elif v<0.8:
            return [0, (0.8-v)/0.2, 1]
        else:
            return [(v-0.8)/0.2, 0, 1]
    
    # Create a Rectangle patch
    for ibm, (box, mask) in enumerate(box_mask_list):
        pt0 = (int(box[1]-box[3]/2), int(box[0]-box[2]/2))
        rect = patches.Rectangle(
            pt0, box[3], box[2],
            linewidth=2, edgecolor='r', facecolor="none",
        )
        plt.gca().add_patch(rect)
        color = np.array(get_color(ibm / len(box_mask_list))).reshape([1,1,-1])
        mask_ = np.zeros([pt0[1]+mask.shape[0], pt0[0]+mask.shape[1], 4])
        for i in range(3):
            mask_[pt0[1]:, pt0[0]:, i] = mask
        mask_[pt0[1]:, pt0[0]:, 3] = alpha * mask
        mask_[pt0[1]:, pt0[0]:, :3] = mask_[pt0[1]:, pt0[0]:, :3] * color
        plt.imshow(mask_)
        
    plt.axis("off")
    return

def plot_lines(batch_lines):
    color_index = ["wall", "door", "window"]
    for l in batch_lines:
        plt.plot([l[0], l[2]], [l[1], l[3]], "C"+str(color_index.index(l[4]))+"-")
    return
