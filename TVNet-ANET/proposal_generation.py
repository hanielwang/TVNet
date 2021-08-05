import sys
from lib.PEM_dataset import VideoDataSet
from lib.loss_function import bmn_loss_func, get_mask
import os
import json
import torch
import torch.nn.parallel
import torch.optim as optim
import tensorflow as tf
import numpy as np
import lib.PEM_opts as opts
from lib.PEM_models import BMN
import pandas as pd


sys.dont_write_bytecode = True
def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data

def BMN_inference(opt):
    model = BMN(opt)
    model = torch.nn.DataParallel(model, device_ids=[0]).cuda()
    checkpoint = torch.load("./models/PEM/BMN_best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    test_loader = torch.utils.data.DataLoader(VideoDataSet(opt, subset="validation"),
                                              batch_size=1, shuffle=False,
                                              num_workers=8, pin_memory=True, drop_last=False)
    tscale = opt["temporal_scale"]
    with torch.no_grad():
        i = 0
        for idx, input_data in test_loader:

            #print (i)
            video_name = test_loader.dataset.video_list[idx[0]]

            input_data = input_data.cuda()
            confidence_map, start, end = model(input_data)

            ###########################################################################################################
            # tdf_end1=pd.read_csv("/mnt/storage/scratch/dm19329/dataset/BSN-output/TEM_results_voting_end_BSN_S15_iter700/"+video_name+".csv")
            # tdf_start1=pd.read_csv("/mnt/storage/scratch/dm19329/dataset/BSN-output/TEM_results_voting_start_BSN_reg/"+video_name+".csv")
            # tdf_end2=pd.read_csv("/mnt/storage/scratch/dm19329/dataset/BSN-output/TEM_results_voting_end_BSN_lw_5/"+video_name+".csv")
            # tdf_start2=pd.read_csv("/mnt/storage/scratch/dm19329/dataset/BSN-output/TEM_results_voting_start_BSN_lw_5/"+video_name+".csv")
            tdf_end1=pd.read_csv("./outputs/VEM_end_L_15_at_based_size9_0728/"+video_name+".csv")
            tdf_start1=pd.read_csv("./outputs/VEM_start_L_15_at_based_size9_0728/"+video_name+".csv")
            tdf_end2=pd.read_csv("./outputs/VEM_end_L_5_at_based_size9_0728/"+video_name+".csv")
            tdf_start2=pd.read_csv("./outputs/VEM_start_L_5_at_based_size9_0728/"+video_name+".csv")

            #tdf_end1=pd.read_csv("/mnt/storage/scratch/dm19329/dataset/BSN-output/TEM_results_CNN_end/"+str(i)+".csv")
            #tdf_start1=pd.read_csv("/mnt/storage/scratch/dm19329/dataset/BSN-output/TEM_results_CNN_start/"+str(i)+".csv")
            i = i+1
            tdf_start = tdf_start1 + tdf_start2# + tdf_start3#np.sqrt(tdf_start1* tdf_start2)# + tdf_start3
            tdf_end = tdf_end1 + tdf_end2# + tdf_end3#np.sqrt(tdf_end1* tdf_end2)# + tdf_end3

            tdf= pd.read_csv("./outputs/TEM_Test/"+video_name+".csv")

            start_scores =tdf_start.start.values[:]#  + 1.6*tdf.start.values[:]#* attention_vid# + 1.5*tdf.start.values[:]#start_scores2
            end_scores = tdf_end.end.values[:]# + 1.6*tdf.end.values[:]#* attention_vid# + 1.5*tdf.end.values[:]#end_scores2

            clr_confidence = (confidence_map[0][1]).detach().cpu().numpy()
            reg_confidence = (confidence_map[0][0]).detach().cpu().numpy()

            ######################################### No select proposlas ############################################
            start_list = []
            end_list = []
            new_props = []

            start_bins=np.zeros(len(start_scores))
            start_bins[[0,-1]]=1
            for idx in range(1,len(start_scores)-1):
                if start_scores[idx]>start_scores[idx+1] and start_scores[idx]>start_scores[idx-1]:
                    start_bins[idx]=1
                if start_scores[idx]>1-opt["thresh"]:
                    start_bins[idx]=1
    
            end_bins=np.zeros(len(end_scores))
            end_bins[[0,-1]]=1
            for idx in range(1,len(end_scores)-1):
                if end_scores[idx]>end_scores[idx+1] and end_scores[idx]>end_scores[idx-1]:
                    end_bins[idx]=1
                if start_scores[idx]>1-opt["thresh"]:
                    end_bins[idx]=1
    
            for j in range(tscale):
                if start_bins[j]==1:
                    start_list.append(j)
                if end_bins[j]==1:
                    end_list.append(j)

            start_scores =tdf_start.start.values[:]/2#  + 1.6*tdf.start.values[:]#* attention_vid# + 1.5*tdf.start.values[:]#start_scores2
            end_scores = tdf_end.end.values[:]/2

            for s in start_list:
                for e in end_list:
                    start_index = s
                    end_index = e
                    if start_index < end_index and  end_index<tscale :
                        xmin_score = start_scores[start_index]+0.8*tdf.start.values[:][start_index]#+1.6*tdf.start.values[:][start_index]
                        xmax_score = end_scores[end_index]+0.8*tdf.end.values[:][end_index]#+1.6*tdf.end.values[:][end_index]
                        xmin = start_index / tscale
                        xmax = end_index / tscale
                        clr_score = clr_confidence[s, e-1]
                        reg_score = reg_confidence[s, e-1]
                        bmn_score = clr_score * reg_score
                        score = xmin_score * xmax_score * bmn_score
                        new_props.append([xmin, xmax, xmin_score, xmax_score, clr_score, reg_score, score])

            new_props = np.stack(new_props)
            #########################################################################

            col_name = ["xmin", "xmax", "xmin_score", "xmax_score", "clr_score", "reg_socre", "score"]
            new_df = pd.DataFrame(new_props, columns=col_name)
            new_df.to_csv("./outputs/candidate_proposals/" + video_name + ".csv", index=False)


if __name__ == '__main__':
    opt = opts.parse_opt()
    opt = vars(opt)
    opt["mode"] = "inference"

    if not os.path.exists("outputs/candidate_proposals"):
        os.makedirs("outputs/candidate_proposals")
    BMN_inference(opt)

