import glob
import os
import cv2 as cv
from cv_utils import Matcher
import net

path = "./input"
imgpaths = sorted(glob.glob(os.path.join(path, '*.*')))[2]

imgpath = imgpaths


source = imgpath.split("/")[-1].split(".")

image = cv.imread(imgpaths)
print(f"sdfjskdf {source}")

pricenet = net.YoloNet(weights='weights/pricelabelnet.pt', conf=0.2, iou=0.45)
productnet = net.YoloNet(weights='weights/product.pt',conf=0.4,iou=0.45)
labels = pricenet.detect(image)
products = productnet.detect(image)

matcher = Matcher(image,labels,products)
matcher.stage_algorithm()
matcher.merge(source=source,visualize=True)