import glob
import os
import cv2 as cv
import argparse
import numpy as np
from cv_utils import Matcher
import net
import warnings

def main(args):

    imgpaths = sorted(glob.glob(os.path.join(args.input_dir, '*.*')))

    for imgpath in imgpaths:
        # store source name
        source = imgpath.split("/")[-1].split(".")
        # load image
        image = cv.imread(imgpath)

        # load model objects
        pricenet = net.YoloNet(weights=args.price_weights, conf=args.price_conf, iou=args.price_iou)
        productnet = net.YoloNet(weights=args.product_weights,conf=args.product_conf,iou=args.product_iou)
        # detect labels
        labels = pricenet.detect(image)
        # detect products
        products = productnet.detect(image)
        # skip img if no labels / products found
        if np.isnan(labels).any():
            warnings.warn("no price labels detected in image {}".format(source[0]))
            continue
        if np.isnan(products).any():
            warnings.warn("no products detected in image {}".format(source[0]))
            continue

        # match labels and products
        matcher = Matcher(image,source,labels,products,args.mode,args.alpha,args.beta)
        matcher.match()
        matcher.merge(visualize=args.visualize,outputdir=args.output_dir)

    cv.destroyAllWindows()

if __name__ == "__main__":
    # argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--price_weights', type=str,default='weights/pricelabelnet.pt',help="path to network weights")
    parser.add_argument('--product_weights', type=str,default='weights/product.pt',help="path to network weights")
    parser.add_argument('--price_conf', type=float, default=0.1,help="confidence threshold for model") #
    parser.add_argument('--product_conf', type=float, default=0.4,help="confidence threshold for model")
    parser.add_argument('--price_iou', type=float, default=0.45,help="iou thresh for model") #
    parser.add_argument('--product_iou', type=float, default=0.45,help="iou thresh for model")
    parser.add_argument('--alpha', type=float, default=2,help="a weight for the matching cost function. The matcher"
                                                                 "computes magnitude of distance vector dt between label and"
                                                                 "product. Further it computes the distance in X direction dtx, and"
                                                                 "weights them by cost = alpha * dtx + dt. ")
    parser.add_argument('--beta', type=float, default=2,help="distance threshold between product and label ")
    parser.add_argument('--input_dir', type=str, default="./input",help="input dir of images")
    parser.add_argument('--output_dir', type=str, default="./output",help="output dir of crops")
    parser.add_argument('--mode', type=str, default="below",help='autolevel,above or below. specifies whether the labels'
                                                                          'are above or below the products. if on auto,'
                                                                          'algo tries to determine it automatically by looking'
                                                                          'at the uppest and lowest label. e.g. if there is no'
                                                                          'product below the lowest label, but there is a product'
                                                                          'above the highest label, then it assumes that all products'
                                                                          'are above its corresponding label.')
    parser.add_argument('--visualize', type=bool, default=False,help="visualize every image while iterating")


    args = parser.parse_args()
    main(args)