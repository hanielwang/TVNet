import sys
import numpy as np
import pandas as pd
import json
import os
from joblib import Parallel, delayed
#import opts_linshi as opts
import lib.PEM_opts as opts

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def get_infer_dict(opt):
    df = pd.read_csv(opt["video_info"])
    json_data = load_json(opt["video_anno"])
    database = json_data
    video_dict = {}
    for i in range(len(df)):
        video_name = df.video.values[i]
        video_info = database[video_name]
        video_new_info = {}
        video_new_info['duration_frame'] = video_info['duration_frame']
        video_new_info['duration_second'] = video_info['duration_second']
        video_new_info["feature_frame"] = video_info['feature_frame']
        video_subset = df.subset.values[i]
        video_new_info['annotations'] = video_info['annotations']
        if video_subset == 'validation':
            video_dict[video_name] = video_new_info
    return video_dict


def Soft_NMS(df, nms_threshold=1e-5, num_prop=200):
    '''
    From BSN code
    :param df:
    :param nms_threshold:
    :return:
    '''
    df = df.sort_values(by="score", ascending=False)

    tstart = list(df.xmin.values[:])
    tend = list(df.xmax.values[:])
    tscore = list(df.score.values[:])

    rstart = []
    rend = []
    rscore = []

    while len(tscore) > 1 and len(rscore) < num_prop and max(tscore)>0:
        max_index = tscore.index(max(tscore))
        for idx in range(0, len(tscore)):
            if idx != max_index:
                tmp_iou = IOU(tstart[max_index], tend[max_index], tstart[idx], tend[idx])
                if tmp_iou > 0:
                    tscore[idx] = tscore[idx] * np.exp(-np.square(tmp_iou) / nms_threshold)

        rstart.append(tstart[max_index])
        rend.append(tend[max_index])
        rscore.append(tscore[max_index])
        tstart.pop(max_index)
        tend.pop(max_index)
        tscore.pop(max_index)

    newDf = pd.DataFrame()
    newDf['score'] = rscore
    newDf['xmin'] = rstart
    newDf['xmax'] = rend
    return newDf


def IOU(s1, e1, s2, e2):
    if (s2 > e1) or (s1 > e2):
        return 0
    Aor = max(e1, e2) - min(s1, s2)
    Aand = min(e1, e2) - max(s1, s2)
    return float(Aand) / Aor

def _gen_detection_video(video_name, video_score, video_cls, video_info, opt, num_prop=200, topk = 2):
    
    score_1 = np.max(video_score)
    class_1 = video_cls[np.argmax(video_score)]
    video_score[np.argmax(video_score)] = -1
    score_2 = np.max(video_score)
    class_2 = video_cls[np.argmax(video_score)]

    df = pd.read_csv("./outputs/candidate_proposals/v_" + video_name + ".csv")
    # if video_name == '--1DO2V4K74':
    #     print (df)
    df['score'] = df.score.values[:]#df.clr_score.values[:] * df.reg_socre.values[:]
    
    if len(df) > 1:
        #df = Soft_NMS(df, 0.52)
        df = Soft_NMS(df, opt["nms_thr"])

    df = df.sort_values(by="score", ascending=False)

    video_duration = video_info["duration_second"]

    proposal_list = []

    for j in range(min(200, len(df))):
        tmp_proposal = {}
        tmp_proposal["label"] = str(class_1)
        tmp_proposal["score"] = float(df.score.values[j] * score_1)
        tmp_proposal["segment"] = [max(0, df.xmin.values[j]) * video_duration,
                                   min(1, df.xmax.values[j]) * video_duration]
        proposal_list.append(tmp_proposal)
    for j in range(min(200, len(df))):
        tmp_proposal = {}
        tmp_proposal["label"] = str(class_2)
        tmp_proposal["score"] = float(df.score.values[j] * score_2)
        tmp_proposal["segment"] = [max(0, df.xmin.values[j]) * video_duration,
                                   min(1, df.xmax.values[j]) * video_duration]
        proposal_list.append(tmp_proposal)

    #print('The video {} is finished'.format(video_name))
    #print (tmp_proposal)
    # if video_name == '--1DO2V4K74':
    #     print (proposal_list)
    return {video_name: proposal_list}

def gen_detection_multicore(opt):
    # get video duration
    infer_dict = get_infer_dict(opt)

    # load class name and video level classification
    cls_data = load_json("./data/cuhk_val_simp_share.json")    # cls_data_score, cls_data_action = cls_data["results"], cls_data["class"]
    cls_data_score, cls_data_cls = {}, {}
    for idx, vid in enumerate(infer_dict.keys()):
        vid = vid[2:]
        cls_data_score[vid] = np.array(cls_data["results"][vid])
        cls_data_cls[vid] = cls_data["class"] #[np.argmax(cls_data_score[vid])] # find the max class



    parallel = Parallel(n_jobs=15, prefer="processes")
    detection = parallel(delayed(_gen_detection_video)(vid, cls_data_score[vid], video_cls, infer_dict['v_'+vid], opt)
                        for vid, video_cls in cls_data_cls.items())
    detection_dict = {}
    #print (detection[0])
    [detection_dict.update(d) for d in detection]
    #detection_dict = dict(detection)
    output_dict = {"version": "ANET v1.3, GTAD", "results": detection_dict, "external_data": {}}

    with open('./outputs/detection_result.json', "w") as out:
        json.dump(output_dict, out)

def main(opt):
    print("Detection post processing start")
    gen_detection_multicore(opt)
    print("Detection Post processing finished")


    print("Evaluation start")
    from Evaluation.eval_detection import ANETdetection

    ground_truth_filename = "./Evaluation/data/activity_net_1_3_new.json"
    prediction_filename = './outputs/detection_result.json'
    subset='validation'
    tiou_thresholds=np.linspace(0.5, 0.95, 10)

    anet_detection = ANETdetection(ground_truth_filename, prediction_filename,
                                   subset=subset, tiou_thresholds=tiou_thresholds,
                                   verbose=True, check_status=False)
    anet_detection.evaluate()


if __name__ == "__main__":
    opt = opts.parse_opt()
    opt = vars(opt)
    main(opt)


