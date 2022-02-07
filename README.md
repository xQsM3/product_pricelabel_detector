# PRODUCT PRICE LABEL DETECTION

Yolov5 Deep Convolutional Network implementation for price label and product detection in shelf stocks

## Author
Luc Stiemer Ba Eng. FH Aachen University of Applied Science luc.stiemer@alumni.fh-aachen.de

## Installation

tested on Ubuntu 20 with RTX 3030 i GPU machine, CUDA 11, python 3.8
pip install requirements

## Usage

python main.py 

optional arguments:
--price_weights
--product_weights
--price_conf'
--product_conf'
--price_iou'# iou thresh for model
-product_iou'# iou thresh for model
--input_dir'
-output_dir'
--mode
--alpha
--beta
'--visualize'


most important argument is --mode, on autolevel, the algo tries to determine whether the labels are above or below the products. if on below / above, labels and products are splitted into level groups (height of shelves) and on "below" it is assumed that (most of) the labels are BELOW products and vise versa. if you are in autolevel mode, watch out for terminal warnings incase the autolevel detection was not succesfull, labels will be assigned to products depending on the minimum distance. not as robust as if the autolevel found a  below / above mode succesfully.

put images in ./input or specify a custom input dir with --input_dir
play around with conf scores and iou threshs incase result is not satisfying. however, defautl worked well for all the images during sanity check.

alpha is the weight in the cost function for matching: cost = dtx * alpha + dt. where dt is the distance vector magnitude between label and prduct, dtx is distance in x direction. THerefore, the
higher alpha, the more it cares for small distances in x direction rather than dt magnitude. 

Beta is just a threshold for maximum distance between label / product, which is allowed to be assigned. Default worked for images during sanity check.

you can visualize the iterations by --visualize True

This algo is the most robust if input images have non interrupted shelve levels and if all labels are either above / below the products. if you face some unrobust samples, you might consider to crop the image such that the mentioned conditions are satisfied.



## Implementation
2 Yolov5s nets 
-->price label detection
-->product detection

2 stage matching algorithm to mach price label and products
preprocess
--> seperate labels and products into shelf levels
1th stage
-->assign products to label which left and right x coordinates bound the label
2th stage
--> assign remaing products which have minimum (weighted) distance to specific label. weighted cost function is cost = dtx * alpha + dt 
where dt is the magnitude of vector betweeen center product and center label. dtx is center distance in x direction.


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


