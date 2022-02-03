import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity
import os
import pandas as pd
import random

## convert cos functions
def xcycwh2xywh(bboxc): # where x y is top left corner acc. to opencv cos
    bbox = bboxc.copy()
    if bbox.ndim == 1:
        bbox[0] = bbox[0] - bbox[2] // 2
        bbox[1] = bbox[1] - bbox[3] // 2
        return bbox
    for i,b in enumerate(bbox):
        bbox[i,0] = b[0]-b[2]//2
        bbox[i,1] = b[1]-b[3]//2
    return bbox

def xyxy2xywh(bboxc):
    bbox = bboxc.copy()
    if bbox.ndim == 1:
        bbox[2] = bbox[2]-bbox[0]
        bbox[3] = bbox[3]-bbox[1]
        return bbox
    for i,b in enumerate(bbox):
        bbox[i,2] = b[2]-b[0]
        bbox[i,3] = b[3]-b[1]
    return bbox

def xywh2xyxy(bboxc):
    bbox = bboxc.copy()
    if bbox.ndim == 1:
        bbox[2] = bbox[0]+bbox[2]
        bbox[3] = bbox[1]+bbox[3]
        return bbox
    for i,b in enumerate(bbox):
        bbox[i,2] = b[0]+b[2]
        bbox[i,3] = b[1]+b[3]
    return bbox

def SSIMsearch(label, y_top, image, direction):
    # SSIM algo searches for similarities between two image crops
    x_step = label[2] // 4
    y_low = label[1]
    y_top = y_top // 2
    if direction == "left":
        d = -1
        x_left = label[0]
        x_right = x_left + x_step
    elif direction == "right":
        d = 1
        x_left = label[0] + label[2] - x_step
        x_right = x_left + x_step

    cropref = image[y_top:y_low, x_left:x_right]


    x_left = x_left + x_step * d
    x_right = x_right + d * x_step
    crop = image[y_top:y_low, x_left:x_right]

    while True:
        if cropref.shape != crop.shape:
            break
        (score, _) = structural_similarity(cropref, crop, full=True, channel_axis=2)

        if score < 0.4:
            break
        #cv.namedWindow(direction, cv.WINDOW_NORMAL)
        #cv.imshow(direction, cropref)
        #cv.waitKey()
        #cv.destroyAllWindows()
        cropref = crop
        x_left = d * x_step + x_left
        x_right = d * x_step + x_right
        crop = image[y_top:y_low, x_left:x_right]

    return int((x_left + x_right + d * x_step) / 2)

def sort_labels_by_level(bboxs): # returns level sorted labels in a dictionary
    bboxs = np.flip(bboxs[bboxs[:, 1].argsort()], axis=0) # sort by "level" respectively y coordinate

    levellabels = {0:{0:bboxs[0]}} # first key is level, second key is label number
    levelheights = {}
    thresh = np.mean(bboxs[:, 3])
    level = 0
    for i,l in enumerate(np.atleast_2d(bboxs)):
        if abs(l[1]-bboxs[i + 1, 1])  < thresh: # stack current label on current level if they are on the same height
            levellabels[level][i+1] = bboxs[i + 1]
        else:
            levelheights[level] = (l[1] + l[3],
                                   bboxs[i + 1, 1] + bboxs[i + 1, 3]) # save height of recent level in (y_low,y_up)
            level += 1 #update level
            levellabels[level] = {i+1:bboxs[i + 1]} # open a new level if current label is higher than the recent once


        if i == bboxs.shape[0]-2: # if no labels above, break and assign final level height to 0 pixel co
            levelheights[level] = (l[1] + l[3],0)
            break

    #return levellabels, a nested dic label number tag and bbox coordinates
    return levellabels,levelheights

def sort_products_by_level(bboxs,levelheights): # returns level sorted products in a dictionary
    bboxs = np.flip(bboxs[bboxs[:, 1].argsort()], axis=0) # sort by "level" respectively y coordinate


    bbox = np.concatenate([bboxs[0],np.array([None])]) # add none as "assigned", to store rather product was assigned to label

    levelbboxs = {}

    for level,heights in levelheights.items():
        bound_low, bound_up = heights
        for i,b in enumerate(np.atleast_2d(bboxs)):
            if b[1] > bound_up and b[1]+b[3] < bound_low:  # stack current label on current level if they are on the same height
                if level in levelbboxs:
                    bbox = np.concatenate([b, np.array([None])])
                    levelbboxs[level] = np.vstack([levelbboxs[level], bbox])
                else:
                    bbox = np.concatenate([b, np.array([None])])
                    levelbboxs[level] = bbox  # open a new level if current label is higher than the recent once

    # returns a level dictionary with
    # bbox array [x,y,w,h,assigned] where assigned is boolean rather product is already assigned to label
    return levelbboxs

class Matcher(): # handles matching price labels with products and cropping
    def __init__(self,img,labels,products):
        self.products = products
        self.levellabels,self.levelheights = sort_labels_by_level(labels)
        self.levelproducts = sort_products_by_level(products,self.levelheights)
        self.img = img

        self.detect_image = img # image showing detections of labels and products
        self.merge_image = img # image showing bboxes of merged label/product groups
        self.crops = [] #list of crops images to be stored

        self.pdcol = ["level", "tag", "lx1","ly1","lx2","ly2","px1","py1","px2","py2"]
        self.matches = pd.DataFrame(columns=self.pdcol)
        # create objects in colums for np array
        #self.matches["assigned_label_co"].astype(object)
        #self.matches["product_co"].astype(object)


        #self.matches = {}

    def stage_algorithm(self):
        bboxs = []
        # stage 1 algorithm
        for level in self.levellabels.keys():
            self.stage1(level)
        # stage 2 algorithm
        for level in self.levellabels.keys():
            self.stage2(level)

    def stage1(self,level):
        # break if no products in this level
        if not level in self.levelproducts:
            return
        # assign all products to label which bound the label on left and right
        for tag,label in self.levellabels[level].items():
            # cordinates label
            lx1,ly1,lx2,ly2 = xywh2xyxy(label)

            for i,product in enumerate(np.atleast_2d(self.levelproducts[level])):

                # x cordinates product
                px1,py1,px2,py2,_ = xywh2xyxy(product)

                bounded = (lx1 > px1 and lx2 < px2) or (lx1 < px1 and lx2 > px2)
                assigned = product[4]

                if bounded and assigned==None:
                    product[4] = tag
                    # pd frame will be useful in matching algorithm
                    match = pd.DataFrame(data=[{"level":level, "tag":tag, "lx1":lx1,"ly1":ly1,
                                                "lx2":lx2,"ly2":ly2,"px1":px1,"py1":py1,
                                                "px2":px2,"py2":py2}])
                    self.matches = self.matches.append(match)

    def stage2(self,level):
        # break if no products in this level
        if not level in self.levelproducts:
            return
        # assigns all products to label which are not assigned yet but are either left or right bounded with label
        for tag,label in self.levellabels[level].items():
            # x cordinates label
            lx1, ly1,lx2,  ly2 = xywh2xyxy(label)
            for i,product in enumerate(np.atleast_2d(self.levelproducts[level])):
                # x cordinates product
                px1, py1,px2, py2,_ = xywh2xyxy(product)
                # bounded left or right
                bounded = (lx1 < px2 and lx2 > px2) or (lx1 < px1 and lx2 > px1)
                assigned = product[4]
                if assigned==None and bounded:
                    product[4] = tag
                    # pd frame will be useful in matching algorithm
                    match = pd.DataFrame(data=[{"level":level, "tag":tag, "lx1":lx1,"ly1":ly1,
                                                "lx2":lx2,"ly2":ly2,"px1":px1,"py1":py1,
                                                "px2":px2,"py2":py2}])

                    self.matches = self.matches.append(match)


    def merge(self,source,visualize=True,outputdir=""): # merges label+assigned products to one bbox
        tag_min = self.matches["tag"].min()
        tag_max = self.matches["tag"].max()

        for tag in range(tag_min,tag_max+1):
            # get all products which are tagged to current label
            group = self.matches.loc[self.matches['tag'] == tag]
            #["level", "tag", "lx1","ly1","lx2","ly2","px1","py1","px2","py2"]
            if group.empty: #if no product tagged on label, continue
                continue
            # get max coordinates from products for embracing bbox
            px1 = group["px1"].min()
            py1 = group["py1"].min()
            px2 = group["px2"].max()
            py2 = group["py2"].max()
            # add max coordinates from label
            lx1 = group["lx1"].min()
            ly1 = group["ly1"].min()
            lx2 = group["lx2"].max()
            ly2 = group["ly2"].max()

            # get max between product / label
            x1 = min(px1,lx1)
            y1 = min(py1,ly1)
            x2 = max(px2,lx2)
            y2 = max(py2,ly2)

            # create bboxes
            embbox = np.array([x1,y1,x2,y2])
            product = np.array([px1,py1,px2,py2])
            label = np.array([lx1,ly1,lx2,ly2])

            # draw bboxes
            r = random.randint(0, 255)
            g = random.randint(0, 255)
            b = random.randint(0, 255)
            color= (r, g, b)

            self.merge_image = cv.rectangle(self.merge_image,tuple(embbox[0:2]),tuple(embbox[2:]),color,6)
            self.merge_image = cv.rectangle(self.merge_image,tuple(product[0:2]),tuple(product[2:]),color,6)
            self.merge_image = cv.rectangle(self.merge_image,tuple(label[0:2]),tuple(label[2:]),color,3)

            # draw centerline between label and product
            centerl = ((label[0]+label[2])//2,(label[1]+label[3])//2)
            centerp = ((product[0]+product[2])//2,(product[1]+product[3])//2)
            self.merge_image = cv.line(self.merge_image,centerl,centerp,color,3)

            # crop
            self.crops.append(self.img[embbox[1]:embbox[3], embbox[0]:embbox[2]])


        # draw detections
        for product in self.products:
            p = xywh2xyxy(product)
            #self.merge_image = cv.rectangle(self.merge_image, (product[0],product[1]), (product[2],product[3]), (0,0,0), 2)
            self.merge_image = cv.rectangle(self.merge_image, tuple(p[0:2]), tuple(p[2:]), (0,0,0), 2)

        if visualize:
            cv.namedWindow("image", cv.WINDOW_NORMAL)
            cv.resizeWindow('image', 800, 800)
            cv.imshow("image",self.merge_image)
            cv.waitKey()
            cv.destroyAllWindows()



        if outputdir != "":
            if not os.path.exists(outputdir):
                os.makedirs(outputdir)

            #save sanity check image
            cv.imwrite(os.path.join(outputdir, "Sanity_Check_Images", f"{source[0]}.{source[1]}"), self.merge_image)
            # save crops
            for i,c in enumerate(self.crops):
                cv.imwrite(os.path.join(outputdir,"crops",f"{source[0]}_PRODUCT{i}.{source[1]}"),c)





    def _scan_product(self,bboxs,height):
        xl = SSIMsearch(bboxs,height,self.img,direction="left")
        xr = SSIMsearch(bboxs,height,self.img,direction="right")
        return xl,xr

