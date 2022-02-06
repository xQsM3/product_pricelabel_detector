# PRODUCT PRICE LABEL DETECTION

Yolov5 Deep Convolutional Network implementation for price label and product detection in shelf stocks

## Installation

tested on Ubuntu 20 with RTX 3030 i GPU machine
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
'--visualize'

## Implementation
2 Yolov5s nets 
-->price label detection
-->product detection

3 stage matching algorithm to mach price label and products
preprocess
--> seperate labels and products into shelf levels
1th stage
-->assign products to label which left and right x coordinates bound the label
2th stage
--> assign remaing products which bound either left or right x
3th stage
-->SSIM similarity algorithm to compute similarity score between left over products and assigned products, append if similar


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## Author
Luc Stiemer Ba Eng. FH Aachen University of Applied Science luc.stiemer@alumni.fh-aachen.de
