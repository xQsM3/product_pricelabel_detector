import glob
import os
import cv2 as cv
import argparse
import sys
from cv_utils import Matcher
import net
import cv_utils
import numpy as np
currentpath = os.path.realpath(__file__)
sys.path.insert(0,"/home/linx123-rtx/product_pricelabel/yolov5")


def main(args):
    input= "/home/linx123-rtx/yolov5/Datasets/SKU110K/custom/images"
    imgpaths = sorted(glob.glob(os.path.join(input, '*.*')))

    for imgpath in imgpaths:
        print(imgpath)
        source = imgpath.split("/")[-1].split(".")

        image = cv.imread(imgpath)
        draw = image.copy()

        pricenet = net.YoloNet(weights=args.price_weights, conf=args.price_conf, iou=args.price_iou)
        #productnet = net.YoloNet(weights=args.product_weights,conf=args.product_conf,iou=args.product_iou)
        labels = pricenet.detect(image)
        #products = productnet.detect(image)

        #matcher = Matcher(image,labels,products)
        #matcher.stage_algorithm()
        #matcher.merge(source=source,visualize=args.visualize,outputdir=args.output_dir)

        output = "/home/linx123-rtx/yolov5/Datasets/SKU110K/custom/"

        if labels.any() == np.nan:
            continue
        if len(labels) == 1:
            continue
        for row in labels:
            try:
                x1 = int(row[0])
                y1 = int(row[1])
                x2 = int(row[0]) + int(row[2])
                y2 = int(row[1]) + int(row[3])

                imwidth = image.shape[1]
                imheight = image.shape[0]

                w = x2 - x1
                h = y2 - y1
                x = x1 + w / 2
                y = y1 + h / 2

                w = w / imwidth
                h = h / imheight
                x = x / imwidth
                y = y / imheight
                cl = 0

                # draw detections
                #p = cv_utils.xywh2xyxy(row)
                draw = cv.rectangle(draw, (x1,y1), (x2,y2), (0,0,0), 2)

                with open(os.path.join(output, "labels", source[0] + ".txt"), "a") as t:
                    row = f"{cl} {x} {y} {w} {h}\n"
                    t.write(row)

            except:
                print("except")

        #cv.namedWindow("image", cv.WINDOW_NORMAL)
        #cv.resizeWindow('image', 1200, 1200)
        #cv.imshow("image",draw)
        k = cv.waitKey(0)


        #if k == 27:  # Esc key to stop
        #    continue


        cv.imwrite(os.path.join(output,"images", source[0] + ".jpg"),image)
    cv.destroyAllWindows()


if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--price_weights', type=str,default='weights/pricelabelnet.pt') #path to network weights
    parser.add_argument('--product_weights', type=str,default='weights/product.pt') #path to network weights
    parser.add_argument('--price_conf', type=float, default=0.6) # confidence threshold for model

    parser.add_argument('--price_iou', type=float, default=0.001) # iou thresh for model


    main(parser.parse_args())