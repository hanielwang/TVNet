import sys
import numpy as np
import pandas as pd
import json
import os
#from joblib import Parallel, delayed
#import opts_linshi as opts
#from gtad_lib import opts



def main(opt):
    print("Detection post processing start")
    #gen_detection_multicore(opt)
    print("Detection Post processing finished")


    print("Evaluation start")
    from eval_detection import ANETdetection

    ground_truth_filename = "/data/activity_net_1_3_new.json"
    prediction_filename = './output/detection_result_BSN_fusion_final_model_test{}.json'


    anet_detection = ANETdetection(ground_truth_filename, prediction_filename,
         subset='validation', tiou_thresholds=np.linspace(0.5, 0.95, 10),
         verbose=True, check_status=True)
    anet_detection.evaluate()

    # from evaluation.eval_detection import ANETdetection
    # anet_detection = ANETdetection(
    #     ground_truth_filename="./evaluation/activity_net_1_3_new.json",
    #     prediction_filename=os.path.join(opt['output'], "detection_result_nms{}.json".format(opt['nms_thr'])),
    #     subset='validation', verbose=True, check_status=False)
    # anet_detection.evaluate()

    # mAP_at_tIoU = [f'mAP@{t:.2f} {mAP*100:.3f}' for t, mAP in zip(anet_detection.tiou_thresholds, anet_detection.mAP)]
    # results = f'Detection: average-mAP {anet_detection.average_mAP*100:.3f} {" ".join(mAP_at_tIoU)}'
    # print(results)
    # with open(os.path.join(opt['output'], 'results.txt'), 'a') as fobj:
    #     fobj.write(f'{results}\n')


if __name__ == "__main__":
    print("Detection post processing start")
    #gen_detection_multicore(opt)
    print("Detection Post processing finished")


    print("Evaluation start")
    from eval_detection import ANETdetection

    ground_truth_filename = "/data/activity_net_1_3_new.json"
    prediction_filename = './output/detection_result_BSN_fusion_final_model_test{}.json'


    anet_detection = ANETdetection(ground_truth_filename, prediction_filename,
         subset='validation', tiou_thresholds=np.linspace(0.5, 0.95, 10),
         verbose=True, check_status=True)
    anet_detection.evaluate()

