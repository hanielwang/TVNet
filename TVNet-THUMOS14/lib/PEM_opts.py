import argparse

def parse_opt():
    parser = argparse.ArgumentParser()
    # Overall settings
    parser.add_argument(
        '--training_lr',
        type=float,
        default=0.00008)
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=1e-4)
    parser.add_argument(
        '--train_epochs',
        type=int,
        default=5)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8)
    parser.add_argument(
        '--step_size',
        type=int,
        default=5)
    parser.add_argument(
        '--step_gamma',
        type=float,
        default=0.1)

    parser.add_argument(
        '--n_gpu',
        type=int,
        default=2)
    parser.add_argument(
        '--n_cpu',
        type=int,
        default=8)

    # output settings
    parser.add_argument('--subset', type=str, default='validation')
    parser.add_argument('--output', type=str, default="./output/PEM")
    parser.add_argument(
        '--video_info',
        type=str,
        default="./data/thumos_annotations/")
    parser.add_argument(
        '--video_anno',
        type=str,
        default="./data/thumos_annotations/")
    parser.add_argument(
        '--temporal_scale',
        type=int,
        default=80)  
    parser.add_argument(
        '--feature_path',
        type=str,
        default="./data/thumos_features/Thumos_feature_hdf5")
    parser.add_argument(
        '--thresh',
        type=int,
        default=0.7)
    parser.add_argument(
        '--feat_ratio',
        type=int,
        default=19.5)
    parser.add_argument(
        '--feat_dim',
        type=int,
        default=2048)

    # anchors
    parser.add_argument('--max_duration', type=int, default=80) 
    parser.add_argument('--min_duration', type=int, default=0) 


    parser.add_argument(
        '--skip_videoframes',
        type=int,
        default=5,
        help='the number of video frames to skip in between each one. using 1 means that there is no skip.'
    )

    # NMS
    parser.add_argument(
        '--nms_thr',
        type=float,
        default=0.55)


    parser.add_argument(
        '--override', default=False, action='store_true',
        help='Prevent use of cached data'
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=1,
        help='for classification'
    )
    parser.add_argument(
        '--ceof',
        type=int,
        default=0,
        help='for classification'
    )
    parser.add_argument(
        '--prop_boundary_ratio',
        type=int,
        default=0.5)
    parser.add_argument(
        '--num_sample',
        type=int,
        default=32)
    parser.add_argument(
        '--num_sample_perbin',
        type=int,
        default=3)


    args = parser.parse_args()

    return args




