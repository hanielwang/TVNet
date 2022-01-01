
import os
import math
import numpy as np
import pandas as pd
import torch.nn.parallel
import json
import lib.PEM_opts as opts
from lib.PEM_models import BMN
from lib.PEM_dataset import VideoDataSet


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    if not os.path.exists("./outputs/candidate_proposals"):
        os.makedirs("./outputs/candidate_proposals")
    model = BMN(opt)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    checkpoint = torch.load("./models/PEM/BMN_best_d=80.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation", mode='inference'),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)
    def load_json(file):
        with open(file) as json_file:
            data = json.load(json_file)
            return data
    ratio = load_json("./data/features_ratio_pro.json")

    with torch.no_grad():
        for idx, input_data in test_loader:
            video_name = test_loader.dataset.video_list[idx[0]]
            offset = min(test_loader.dataset.data['indices'][idx[0]])
            tem_score = pd.read_csv("./outputs/TEM_Test/" + video_name + ".csv")
            ratio_vid = float(ratio[video_name][0])
            tdf_start01=pd.read_csv("./outputs/VEM_Test/L10_start/"+video_name+".csv")
            tdf_end01=pd.read_csv("./outputs/VEM_Test/L10_end/"+video_name+".csv")
            tdf_start02=pd.read_csv("./outputs/VEM_Test/L5_start/"+video_name+".csv")
            tdf_end02=pd.read_csv("./outputs/VEM_Test/L5_end/"+video_name+".csv") 
 
            frame_list = tem_score.frame.values[:]
            tem_start_scores = tem_score.start.values[:]
            tem_end_scores = tem_score.end.values[:]
            len_tem = len(tem_score.end.values[:])
            video_name = video_name+'_{}'.format(math.floor(offset/160))

            tdf_start = (tdf_start01 +tdf_start02)
            tdf_end = (tdf_end01 +tdf_end02)

            input_data = input_data.cuda()
            confidence_map, start, end = model(input_data)

            start_scores = tdf_start.start.values[:]
            end_scores = tdf_end.end.values[:]
            len_vid = len(start_scores)

            clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
            reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()


            ################################### select starting/ending ###########################################
            #print("select candidate points start...")
            start_list = []
            end_list = []
            new_props = []
            tscale =len(start_scores)

            start_bins=np.zeros(len(start_scores))
            start_bins[[0,-1]]=1
            for idx in range(1,len(start_scores)-1):
                if start_scores[idx]>start_scores[idx+1] and start_scores[idx]>start_scores[idx-1]:
                    start_bins[idx]=1
                if start_scores[idx]>1-opt['thresh']:
                    start_bins[idx]=1
    
            end_bins=np.zeros(len(end_scores))
            end_bins[[0,-1]]=1
            for idx in range(1,len(end_scores)-1):
                if end_scores[idx]>end_scores[idx+1] and end_scores[idx]>end_scores[idx-1]:
                    end_bins[idx]=1
                if start_scores[idx]>1-opt['thresh']:
                    end_bins[idx]=1
    
            for j in range(tscale):
                if start_bins[j]==1:
                    start_list.append(j)
                if end_bins[j]==1:
                    end_list.append(j)

            start_list_rescale = [int(ratio_vid*d/opt['skip_videoframes']) for d in start_list]
            end_list_rescale = [int(ratio_vid*d/opt['skip_videoframes']) for d in end_list]

            n_now_s = []
            for n, value in enumerate(start_list_rescale):
                if value >= offset/opt['skip_videoframes'] and value <= (offset/opt['skip_videoframes']+opt["max_duration"]):
                    n_now_s.append(n)

            n_now_e = []
            for n, value in enumerate(end_list_rescale):
                if value >= offset/opt['skip_videoframes'] and value <= (offset/opt['skip_videoframes']+opt["max_duration"]):
                    n_now_e.append(n)

            start_list_final = np.array(start_list_rescale)[n_now_s]
            end_list_final = np.array(end_list_rescale)[n_now_e]
 
            start_list_final = [int(m - (offset/opt['skip_videoframes'])) for m in start_list_final]
            end_list_final = [int(m - (offset/opt['skip_videoframes'])) for m in end_list_final]

            start_scores = tdf_start.start.values[:]/2
            end_scores = tdf_end.end.values[:]/2

            ######################################## generate proposals and add PEM scores ########################################
            #print("proposals generation and calculate confidence scores start...")
            new_props = []
            for idx in start_list_final:
                for jdx in end_list_final:
                    start_index = idx
                    end_index = jdx
                    if start_index < end_index and  end_index<opt["temporal_scale"] :
                        xmin = start_index * opt['skip_videoframes'] + offset
                        xmax = end_index * opt['skip_videoframes'] + offset
                        s_feat_idx = min(int(xmin/ratio_vid),int(len_vid-1))
                        e_feat_idx = min(int(xmax/ratio_vid),int(len_vid-1))
                        tem_s_idx = min(int(start_index + offset/opt['skip_videoframes']),len_tem-1)
                        tem_e_idx = min(int(end_index + offset/opt['skip_videoframes']),len_tem-1)
                        xmin_score = start_scores[s_feat_idx] + 0.6*tem_start_scores[tem_s_idx]
                        xmax_score = end_scores[e_feat_idx] + 0.6*tem_end_scores[tem_e_idx]
                        clr_score = clr_confidence[idx, jdx-1]
                        reg_score = reg_confidence[idx, jdx-1]
                        pem_score = clr_score * reg_score
                        score = xmin_score * xmax_score * pem_score
                        new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])
            if new_props:
                new_props = np.stack(new_props)
                col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_socre", "score"]
                new_df = pd.DataFrame(new_props, columns=col_name)
                new_df.to_csv("./outputs/candidate_proposals/" + video_name + ".csv", index=False)

    print("Got candidate proposals")
