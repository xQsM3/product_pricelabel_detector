import glob
import os
import cv2 as cv
import argparse

from cv_utils import Matcher
import net


def main(args):

    imgpaths = sorted(glob.glob(os.path.join(args.input_dir, '*.*')))

    for imgpath in imgpaths:
        source = imgpath.split("/")[-1].split(".")

        image = cv.imread(imgpath)


        pricenet = net.YoloNet(weights=args.price_weights, conf=args.price_conf, iou=args.price_iou)
        productnet = net.YoloNet(weights=args.product_weights,conf=args.product_conf,iou=args.product_iou)
        labels = pricenet.detect(image)
        products = productnet.detect(image)

        matcher = Matcher(image,labels,products)
        matcher.stage_algorithm()
        matcher.merge(source=source,visualize=args.visualize)

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--price_weights', type=str,default='weights/pricelabelnet.pt') #path to network weights
    parser.add_argument('--product_weights', type=str,default='weights/product.pt') #path to network weights
    parser.add_argument('--price_conf', type=float, default=0.2) # confidence threshold for model
    parser.add_argument('--product_conf', type=float, default=0.4)
    parser.add_argument('--price_iou', type=float, default=0.45) # iou thresh for model
    parser.add_argument('--product_iou', type=float, default=0.45) # iou thresh for model
    parser.add_argument('--input_dir', type=str, default="./input")
    parser.add_argument('--output_dir', type=str, default="./output")
    parser.add_argument('--visualize', type=bool, default="True") # visualize every image while iterating


    args = parser.parse_args()
    main(args)