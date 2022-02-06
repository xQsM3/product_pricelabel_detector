
import cv2 as cv
import numpy as np
from skimage.metrics import structural_similarity
import os
import pandas as pd
import random
import warnings
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

def SSIMsearch(ref, candidate, image):
    # SSIM algo searches for similarities between two image crops
    x1,y1,x2,y2 = xywh2xyxy(ref)
    cropref = image[y1:y2, x1:x2]
    x1, y1, x2, y2 = xywh2xyxy(candidate)
    cropcan = image[y1:y2, x1:x2]


    (score, _) = structural_similarity(cropref, cropcan, full=True, channel_axis=2)

    return score

def sort_labels_by_level(bboxs,label_position,img): # returns level sorted labels in a dictionary
    bboxs = np.flip(bboxs[bboxs[:, 1].argsort()], axis=0) # sort by "level" respectively y coordinate

    levellabels = {0:{0:bboxs[0]}} # first key is level, second key is label number
    levelheights = None
    thresh = np.mean(bboxs[:, 3])
    level = 0


    # dont enter for loop if only one label exists
    if bboxs.ndim == 1:
        if label_position == "below":
            levelheights = np.array([bboxs[0,1],0])
        elif label_position == "above":
            levelheights = np.array([img.shape[1], bboxs[0,1]])
        return levellabels, levelheights


    for i,l in enumerate(np.atleast_2d(bboxs)):
        if abs(l[1]-bboxs[i + 1, 1])  < thresh: # stack current label on current level if they are on the same height
            levellabels[level][i+1] = bboxs[i + 1]
        else:
            height = np.array([l[1] + l[3],bboxs[i + 1, 1] + bboxs[i + 1, 3]])
            levelheights = (np.vstack((levelheights, height)) if (levelheights is not None) else height)
            level += 1 #update level
            # open a new level if current label is higher than the recent once
            levellabels[level] = {i+1:bboxs[i + 1]}

        # if no labels above and label position == below, break and assign final level height to 0 pixel co
        if i >= bboxs.shape[0]-2 and (label_position == "below" or label_position == "undefined"):
            try:
                height = np.array([np.atleast_2d(levelheights)[-1,1],0])
                levelheights = (np.vstack((levelheights, height)) if (levelheights is not None) else height)
            except: # levelheight is None
                levelheights = np.array([img.shape[1],0])
            break
        # if no labels below and label position == above insert new level 0 and break
        elif i >= bboxs.shape[0]-2 and label_position == "above":
            try:
                height = np.array([img.shape[1], np.atleast_2d(levelheights)[0,1]])
                levelheights = (np.vstack((height,levelheights)) if (levelheights is not None) else height)
            except: # levelheights is none
                levelheights = np.array([img.shape[1], 0])
            break


    #return levellabels, a nested dic label number tag and bbox coordinates
    return levellabels,levelheights

def sort_products_by_level(bboxs,levelheights): # returns level sorted products in a dictionary
    bboxs = np.flip(bboxs[bboxs[:, 1].argsort()], axis=0) # sort by "level" respectively y coordinate

    # stack to np.nan columns on levelbox, one for level and one for label assignment
    levelbboxs = bboxs
    e = np.empty((levelbboxs.shape[0], 1))
    e[:] = np.NaN
    levelbboxs= np.hstack((levelbboxs, e,e))

    # assign level to the boxes
    for level, heights in enumerate(np.atleast_2d(levelheights)):
        for levelbbox in levelbboxs:
            if levelbbox[1]>heights[1] and levelbbox[1]+levelbbox[3] < heights[0]:
                if levelbbox[1] > heights[1] and levelbbox[1] + levelbbox[3] < heights[0]:
                    levelbbox[5] = level
    # returns a level dictionary with
    # bbox array [x,y,w,h,assigned] where assigned is boolean rather product is already assigned to label
    return levelbboxs

class Matcher(): # handles matching price labels with products and cropping
    def __init__(self, img, source, labels, products, mode,alpha,beta, SSIMthresh=0.7):
        self.products = products
        self.labels = labels
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.img = img.copy()
        self.SSIMthresh = SSIMthresh
        self.source = source
        self.merge_image = img.copy() # image showing bboxes of merged label/product groups
        self.crops = [] #list of crops images to be stored

        # split labels and products into height levels if possible
        if self.mode == "autolevel":
            self.auto_label_position()

        #if self.mode != "undefined":
        self.levellabels,self.levelheights = sort_labels_by_level(labels, self.mode, img)
        self.levelproducts = sort_products_by_level(products,self.levelheights)


    def auto_label_position(self):
        # get highest label and lowest label
        highest_label = np.amin(self.labels,axis=(0,1))
        lowest_label = np.amax(self.labels,axis=(0,1))
        # get highest and lowest product
        highest_pruduct = np.amin(self.products,axis=(0,1))
        lowest_product = np.amax(self.products,axis=(0,1))

        # if highest label is under highest product and lowest level is under lowest product, assume that all labels in
        # image are below corresponding product
        if highest_label > highest_pruduct and lowest_label > lowest_product:
            self.mode = "below"
            print(f"INFO: (autolevel) setting label level mode to below for image {self.source[0]}")
        elif highest_label > highest_pruduct and lowest_label > lowest_product:
            self.mode = "above"
            print(f"INFO: (autolevel) setting label level mode to above for image {self.source[0]}")
        else:
            warnings.warn("auto level not succesfull for image {0}, if output is nonsense, switch mode to below / above.".format(self.source[0]))
            self.mode = "undefined"

    def match(self):
        self.stage_algorithm()

    def stage_algorithm(self):
        # stage 1 algorithm (bounded)
        for level in self.levellabels.keys():
            self.stage1(level)
        # stage 2 algorithm (matches left overs, ignoring detected levels)
        self.stage2()

    def stage1(self,level):
        # break if no products in this level
        if not level in self.levelproducts[:,5]:
            return

        # assign all products to label which bound the label on left and right
        for tag,label in self.levellabels[level].items():
            # cordinates label
            lx1,ly1,lx2,ly2 = xywh2xyxy(label)

            for i,product in enumerate(np.atleast_2d(self.levelproducts)):
                if product[5] != level:
                    continue
                # x cordinates product
                px1,py1,px2,py2,_,_ = xywh2xyxy(product)

                bounded = (lx1 > px1 and lx2 < px2) or (lx1 < px1 and lx2 > px2)
                notassigned = np.isnan(product[4])

                xlc = label[0] + label[2] // 2
                ylc = label[1] + label[3] // 2
                xpc = product[0] + product[2] // 2
                ypc = product[1] + product[3] // 2

                if self.mode == "below":  # find products ABOVE label
                    if ypc > ylc:
                        continue

                elif self.mode == "above":  # find products BELOW label
                    if ypc < ylc:
                        continue

                if bounded and notassigned:
                    product[4] = tag

    def stage2(self):

        candidates = None
        # find label candidates with clostest distance to product
        for i, pt in enumerate(self.levelproducts):

            # continue if alread tagged
            if not np.isnan(pt[4]):
                continue

            # search for label which has smallest distance (vector) to current product
            dt = np.linalg.norm(np.array([self.img.shape[0], self.img.shape[1]]))  # init distance vector
            dtx = self.img.shape[1]
            # cost function
            cost = (dtx * self.alpha + dt) / self.alpha

            for labels in self.levellabels.values():

                for tag, label in labels.items():
                    # get all products which are tagged to current label
                    group = self.levelproducts[self.levelproducts[:, 4] == tag]
                    if len(group) != 0:  # if label is tagged, continue
                        continue

                    xlc = label[0] + label[2] // 2
                    ylc = label[1] + label[3] // 2
                    xpc = pt[0] + pt[2] // 2
                    ypc = pt[1] + pt[3] // 2

                    if self.mode == "below": # find products ABOVE label
                        if ypc > ylc:
                            continue

                    elif self.mode == "above": # find products BELOW label
                        if ypc< ylc:
                            continue

                    vector = np.array([xlc - xpc, ylc - ypc])
                    dt = np.linalg.norm(vector)
                    dtx = abs(xlc - xpc)

                    # cost function
                    newcost = (dtx * self.alpha + dt) / self.alpha

                    thresh = (label[2]+label[3]) *self.beta*self.alpha # arbritary thresh how far away clostest candidate is allowd to be

                    if newcost < cost and newcost < thresh:

                        cost = newcost
                        j = i  # save current best product candidate
                        candidates_tag = tag
                        candidates_cost = newcost

            if "j" in locals():
                candidate = np.hstack([candidates_cost,candidates_tag,j])
                candidates= (np.vstack((candidates, candidate)) if (candidates is not None) else candidate)
                del j

        # tag immediately if only one candidate exist
        if candidates is not None and candidates.ndim == 1:
            index = int(candidates[2])
            self.levelproducts[index, 4] = int(candidates[1])
        # compare candidate distances if more than one exists
        elif candidates is not None:
            self.compare_candidates(candidates)

    def compare_candidates(self,candidates):
        for labels in self.levellabels.values():
            for tag,label in labels.items():
                # skip if label is tagged
                group = self.levelproducts[self.levelproducts[:, 4] == tag]
                if len(group) != 0:  # if label is tagged, continue
                    continue


                # get candidate proposals for current label
                current_candidates = candidates[candidates[:,1]==tag]
                if len(current_candidates) == 0:
                    continue
                # if only one candidate, tag it and continue
                elif current_candidates.ndim == 1:
                    index = int(current_candidates[2] )
                    self.levelproducts[index, 4] = tag
                    continue

                # extract clostest candidate
                closest_candidate = current_candidates[np.argmin(current_candidates[:,0],axis=0)]
                # skip if no product found for thsi label
                if len(closest_candidate) == 0:
                    continue

                index = int(closest_candidate[2])

                # tag closest product to label
                self.levelproducts[index,4] = tag


    def merge(self,visualize=True,outputdir=""): # merges label+assigned products to one bbox


        for level,labels in self.levellabels.items():
            for tag,label in labels.items():
                # get all products which are tagged to current label
                group = self.levelproducts[self.levelproducts[:,4]==tag]
                #["level", "tag", "lx1","ly1","lx2","ly2","px1","py1","px2","py2"]
                if len(group) == 0: #if no product tagged on label, continue
                    continue

                group = xywh2xyxy(group)
                label = xywh2xyxy(label)
                # get max coordinates from products for embracing bbox
                px1 = np.min(group,axis=0)[0]#group["px1"].min()
                py1 = np.min(group,axis=0)[1]#group["px1"].min()

                px2 = np.max(group,axis=0)[2]#group["px1"].min()
                py2 = np.max(group,axis=0)[3]#group["px1"].min()

                lx1 = label[0]#group["px1"].min()
                ly1 = label[1]#group["px1"].min()

                lx2 = label[2]#group["px1"].min()
                ly2 = label[3]#group["px1"].min()

                # get max between product / label
                x1 = min(px1,lx1)
                y1 = min(py1,ly1)
                x2 = max(px2,lx2)
                y2 = max(py2,ly2)

                # create bboxes
                embbox = np.array([x1,y1,x2,y2],dtype=int)
                product = np.array([px1,py1,px2,py2],dtype=int)
                label = np.array([lx1,ly1,lx2,ly2],dtype=int)

                # draw bboxes
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                color= (r, g, b)


                self.merge_image = cv.rectangle(self.merge_image,tuple(embbox[0:2]),tuple(embbox[2:]),color,6)

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
            self.merge_image = cv.rectangle(self.merge_image, tuple(p[0:2]), tuple(p[2:]), (255,0,0), 2)

        # draw labels
        for labels in self.labels:
            l = xywh2xyxy(labels)
            #self.merge_image = cv.rectangle(self.merge_image, (product[0],product[1]), (product[2],product[3]), (0,0,0), 2)
            self.merge_image = cv.rectangle(self.merge_image, tuple(l[0:2]), tuple(l[2:]), (255,0,0), 2)

        if visualize:
            cv.namedWindow("image", cv.WINDOW_NORMAL)
            cv.resizeWindow('image', 800, 800)
            cv.imshow("image",self.merge_image)
            cv.waitKey()




        if outputdir != "":
            if not os.path.exists(os.path.join(outputdir, "sanity_check_images")):
                os.makedirs(os.path.join(outputdir, "sanity_check_images"))

            if not os.path.exists(os.path.join(outputdir, "crops")):
                os.makedirs(os.path.join(outputdir, "crops"))

            #save sanity check image
            cv.imwrite(os.path.join(outputdir, "sanity_check_images", f"{self.source[0]}.{self.source[1]}"), self.merge_image)
            # save crops
            for i,c in enumerate(self.crops):
                cv.imwrite(os.path.join(outputdir,"crops",f"{self.source[0]}_PRODUCT{i}.{self.source[1]}"),c)

