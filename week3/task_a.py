"""
Apply pre-trained MaskRCNN or FasterRCNN on COCO in Out-Of-Context Dataset
"""
import os
import glob
import cv2

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

model_id = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
OOC_path = '../../data/out_of_context/'

if __name__ == "__main__":

    # CONFIGURATION
    # Model config
    cfg = get_cfg()

    # Run a model in detectron2's core library: get file and weights
    cfg.merge_from_file(model_zoo.get_config_file(model_id))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model_id)

    # Hyper-params
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # threshold used to filter out low-scored bounding boxes in predictions
    cfg.MODEL.DEVICE = "cuda"
    cfg.OUTPUT_DIR = 'output'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    predictor = DefaultPredictor(cfg)   # Initialize predictor

    os.makedirs('OOC_inference', exist_ok=True)
    # Iterate through all the images of the dataset
    for idx, img_path in enumerate(sorted(glob.glob(f'{OOC_path}/*.jpg'))):
        im = cv2.imread(img_path)

        outputs = predictor(im)

        v = Visualizer(im[:, :, ::-1],
                       metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))

        out = v.draw_instance_predictions(outputs["instances"].to('cpu'))
        cv2.imwrite(f'OOC_inference/{idx}.png', out.get_image()[:, :, ::-1])