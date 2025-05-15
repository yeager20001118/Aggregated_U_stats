# MMD_AU'       MMDFuse       MMDAgg        MMD-D         AutoTST       MEmabid  
# 0.861 ± 0.027 0.609 ± 0.067 0.364 ± 0.050 0.190 ± 0.034 0.452 ± 0.011 0.226 ± 0.047
# 0.917 ± 0.012 0.704 ± 0.055 0.388 ± 0.066 0.243 ± 0.036 0.563 ± 0.014 0.377 ± 0.055
# 1.000 ± 0.000 0.737 ± 0.057 0.411 ± 0.063 0.378 ± 0.040 0.710 ± 0.140 0.527 ± 0.074
# 1.000 ± 0.000 0.851 ± 0.047 0.440 ± 0.074 0.606 ± 0.052 0.757 ± 0.133 0.722 ± 0.066
# 1.000 ± 0.000 0.935 ± 0.018 0.488 ± 0.051 0.772 ± 0.050 0.800 ± 0.126 0.847 ± 0.028
# 1.000 ± 0.000 0.977 ± 0.015 0.538 ± 0.051 0.809 ± 0.062 0.870 ± 0.095 0.904 ± 0.047


# '--is_cov_tr_MMD_AU' = False '--is_cov_MMD_AU' = True
# 0.861 ± 0.027 0.824 ± 0.029 0.000 ± 0.000 0.000 ± 0.000
# 0.917 ± 0.012 0.896 ± 0.016 0.000 ± 0.000 0.000 ± 0.000
# 1.000 ± 0.000 1.000 ± 0.000 0.000 ± 0.000 0.000 ± 0.000
# 1.000 ± 0.000 1.000 ± 0.000 0.000 ± 0.000 0.000 ± 0.000
# 1.000 ± 0.000 1.000 ± 0.000 0.000 ± 0.000 0.000 ± 0.000
# 1.000 ± 0.000 1.000 ± 0.000 0.000 ± 0.000 0.000 ± 0.000

#0.917 ± 0.012 0.896 ± 0.016 seed 292
#0.983 ± 0.004 0.978 ± 0.006 seed 263

# '--is_cov_tr_MMD_AU' = False '--is_cov_MMD_AU' = False
# 0.777 ± 0.033 0.776 ± 0.033 0.000 0.000 0.000 0.000
# 0.942 ± 0.009 0.942 ± 0.009 0.000 0.000 0.000 0.000
# 0.970 ± 0.017 0.970 ± 0.017 0.000 0.000 0.000 0.000
# 0.996 ± 0.002 0.996 ± 0.002 0.000 0.000 0.000 0.000
# 1.000 ± 0.000 1.000 ± 0.000 0.000 0.000 0.000 0.000
# 1.000 ± 0.000 1.000 ± 0.000 0.000 0.000 0.000 0.000

import numpy as np
import torch
import argparse
import sys
import os
import time
import pickle
# Extend the module search path to include parent directories.
sys.path.append(os.path.abspath('/data/gpfs/projects/punim2335/baselines'))
start_time = time.time()

# Set up command-line argument parser.
parser = argparse.ArgumentParser()

# Experiment configuration parameters.
parser.add_argument('--name',               default='MNIST',                 help='Dataset name')
# Accept multiple sample sizes (one for each experimental condition)
parser.add_argument('--check',              default=1,                       help='Indicator: 1 for test power, 0 for Type-I error')
parser.add_argument('--n_exp',              default=10,                      help='Number of experiment repeats')
parser.add_argument('--n_test',             default=100,                     help='Number of two-sample test executions')
parser.add_argument('--n_per',              default=100,                     help='Number of permutations for bootstrap replications')
parser.add_argument('--alpha',              default=0.05,                    help='Significance level for tests')

# parser.add_argument('--rs',                 default=[189,                292,                383,                483,                583,                683],          help='Base random seed')

parser.add_argument('--rs',                 default=[189,                263,                480,                470,                583,                683],          help='Base random seed')
parser.add_argument('--N1',                 default=[20,                 30,                 40,                 50,                 60,                 70],           help='Sample sizes for each experimental condition')
parser.add_argument('--is_cov_tr_MMD_AU',   default=False,                          help='Flag to use training covariance in the testing procedure of MMD_AU (True/False)')
parser.add_argument('--way',                default=['Fuse', 'Fuse', 'Fuse', 'Fuse', 'Fuse', 'Fuse'],    help="Bandwidth search strategy for each kernel type (e.g. 'Grid', 'Boost')")


# Parameters specific to the MMD_AU method in training phase.
parser.add_argument('--lr_MMD_AU',          default=[0.00005,            0.00005,            0.00005,            0.00005,            0.00005,            0.00005],      help='Learning rate for MMD_AU optimization')
# Bandwidth parameters: expects four sub-lists (one per experimental condition) each with four integers.
parser.add_argument('--n_bandwidth_MMD_AU', default=[[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], help='Bandwidth settings for kernels as list of lists: (gaussian, laplacian, deep_gaussian, deep_laplacian)')
parser.add_argument('--ne_MMD_AU',          default=[1000,               1000,               1000,               1000,               1000,               1000],          help='Number of training epochs for MMD_AU')
parser.add_argument('--bs_MMD_AU',          default=[128,                128,                128,                128,                128,                128],           help='Batch size for MMD_AU training')
parser.add_argument('--is_cov_MMD_AU',      default=[True,               True,               True,               True,               True,               True],         help='Flag to use covariance in MMD_AU (True/False)')


# parameters of C2ST
parser.add_argument('--channels',  default=1,    help='channels of data')
parser.add_argument('--img_size',  default=32,   help='img_size of data')
parser.add_argument('--xout_C2ST', default=100,   help='Output dimension of C2ST')
parser.add_argument('--ne_C2ST', default=1000,   help='Number of C2ST optimization epochs')
parser.add_argument('--bs_C2ST', default=10,    help='Batch size of C2ST in optimization')
parser.add_argument('--lr_C2ST', default=0.0005, help='Learning rate of C2ST in optimization')

# parameters of MMD_D
parser.add_argument('--xin_MMD_D',  default=2,    help='Input dimension of MMD_D')
parser.add_argument('--H_MMD_D',    default=50,   help='Hidden layer dimension of MMD_D')
parser.add_argument('--xout_MMD_D', default=50,   help='Output dimension of MMD_D')
parser.add_argument('--ne_MMD_D', default=1000,   help='Number of MMD_D optimization epochs')
parser.add_argument('--bs_MMD_D', default=256,    help='Batch size of MMD_D in optimization')
parser.add_argument('--lr_MMD_D', default=0.0005, help='Learning rate of MMD_D in optimization')

# parameters of MEmabid
parser.add_argument('--tl_MEmabid',   default=1,     help='Number of test locations of MEmabid')
parser.add_argument('--ne_MEmabid',   default=1000,   help='Number of MEmabid optimization epochs')
parser.add_argument('--bs_MEmabid',   default=256,    help='Batch size of MEmabid in optimization')
parser.add_argument('--lr_MEmabid',   default=0.01, help='Learning rate of MEmabid in optimization')
parser.add_argument('--beta_MEmabid', default=1,      help='Beta of MEmabid in two-sample test')


args = parser.parse_args()

# Determine the number of experimental conditions.
n_conditions = len(args.N1)
# For MMD_AU, record two versions: with and without selection inference.
# However, since you also call TST_MMDFuse and TST_MMDAgg, we'll store outputs for 4 versions.
n_versions = 7
# Initialize result arrays.
Results   = np.zeros((n_conditions, n_versions, int(args.n_exp)))
Results_P = np.zeros_like(Results)
# Arrays to store aggregated (final) results: [mean, standard error].
Final_results   = np.zeros((n_conditions, n_versions, 2))
Final_results_P = np.zeros_like(Final_results)

# # Set up an encoder based on TorchVision's ResNet18 for MMD_AU.
from torchvision.models import resnet18
import torch.nn as nn
np.random.seed(1102)
torch.manual_seed(1102)
torch.cuda.manual_seed_all(1102)  # 如果有GPU
class TorchVisionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(pretrained = True)
        # Remove the final fully connected layer.
        self.backbone.fc = nn.Identity() 
    def forward(self, x):
        x = x.resize(len(x), 1, 32, 32)
        x = x.repeat(1, 3, 1, 1)
        return self.backbone(x)

def check_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

# Process each experimental condition.
for dd in range(n_conditions):
    # try:
    #     with open("model/"  + args.name + "_" + str(args.n_bandwidth_MMD_AU[dd]) + "_"  + str(args.is_cov_MMD_AU[dd]) + "_"  + str(args.N1[dd]) + "_"  + str(args.rs[dd])  + "_"  + str(args.ne_MMD_AU[dd]) + "_"  + str(args.bs_MMD_AU[dd]) + "_"  + str(args.lr_MMD_AU[dd])  + ".pkl", "rb") as f:
    #         model_au = pickle.load(f)
    # except:
    #     encoder = TorchVisionEncoder().to(check_device())
    #     for param in encoder.parameters():
    #         param.requires_grad = True
    #     encoder.train()
    #     from baseline_TST.MMD_AU import train_MMD_AU
    #     model_au = train_MMD_AU(
    #                 name=args.name,
    #                 N1=args.N1[dd],
    #                 rs=args.rs[dd],
    #                 check=args.check,
    #                 N_epoch=args.ne_MMD_AU[dd],
    #                 batch_size=args.bs_MMD_AU[dd],
    #                 learning_rate=args.lr_MMD_AU[dd],
    #                 n_bandwidth=args.n_bandwidth_MMD_AU[dd],
    #                 way=args.way,
    #                 is_cov=args.is_cov_MMD_AU[dd],
    #                 encoder=encoder
    #             )
    #     with open("model/"  + args.name + "_" + str(args.n_bandwidth_MMD_AU[dd]) + "_"  + str(args.is_cov_MMD_AU[dd]) + "_"  + str(args.N1[dd]) + "_"  + str(args.rs[dd])  + "_"  + str(args.ne_MMD_AU[dd]) + "_"  + str(args.bs_MMD_AU[dd]) + "_"  + str(args.lr_MMD_AU[dd])  + ".pkl", "wb") as f:
    #         pickle.dump(model_au, f)

    for kk in range(int(args.n_exp)):
        # Preallocate arrays for each method's raw output for the current condition.
        H_sel    = np.zeros(args.n_test)  # For rejection results of TST_MMD_AU with selection inference.
        H_no_sel = np.zeros(args.n_test)  # For rejection results of TST_MMD_AU without selection inference.
        H_Fuse   = np.zeros(args.n_test)  # For rejection results of TST_MMDFuse.
        H_Agg    = np.zeros(args.n_test)  # For rejection results of TST_MMDAgg.
        H_C2ST_L = np.zeros(args.n_test)  
        H_MMD_D   = np.zeros(args.n_test)  
        H_MEmabid    = np.zeros(args.n_test)  
        
        P_sel    = np.zeros(args.n_test)  # For p-values of TST_MMD_AU with selection inference.
        P_no_sel = np.zeros(args.n_test)  # For p-values of TST_MMD_AU without selection inference.
        P_Fuse   = np.zeros(args.n_test)  # For p-values of TST_MMDFuse.
        P_Agg    = np.zeros(args.n_test)  # For p-values of TST_MMDAgg.
        
        # # ----- TST_MMD_AU Method -----
        # from baseline_TST.MMD_AU import TST_MMD_AU
        # H_sel, H_no_sel, P_sel, P_no_sel, training_time_MMD_AU, testing_time_MMD_AU = TST_MMD_AU(
        #     name=args.name,
        #     N1=args.N1[dd],
        #     rs=kk + args.rs[dd],
        #     check=args.check,
        #     n_test=args.n_test,
        #     n_per=args.n_per,
        #     alpha=args.alpha,
        #     is_cov_tr=args.is_cov_tr_MMD_AU,
        #     model_au=model_au
        # )
        
        # # ----- TST_MMDFuse Method -----
        # from baseline_TST.MMDfuse import TST_MMDFuse
        # H_Fuse, P_Fuse, training_time_Fuse, testing_time_Fuse = TST_MMDFuse(
        #     args.name,
        #     args.N1[dd],
        #     kk + args.rs[dd],
        #     args.check,
        #     args.n_test,
        #     20*args.n_per,
        #     args.alpha
        # )
        # # ----- TST_MMDAgg Method -----
        # from baseline_TST.MMDAgg import TST_MMDAgg
        # H_Agg, P_Agg, training_time_Agg, testing_time_Agg = TST_MMDAgg(
        #     args.name,
        #     args.N1[dd],
        #     kk + args.rs[dd],
        #     args.check,
        #     args.n_test,
        #     5*args.n_per,
        #     args.alpha
        # )

        # # # C2ST-L
        # from baseline_TST.C2ST import TST_C2ST_D
        # H_C2ST_L, _, _ = TST_C2ST_D(args.name, args.N1[dd], kk+args.rs[dd], args.check, args.n_test, args.n_per, args.alpha, check_device(),  torch.float, args.channels, args.img_size, args.xout_C2ST, args.ne_C2ST, args.bs_C2ST, args.lr_C2ST)
        # print('C2ST-L Done!')

        # # # MMD-D
        # from baseline_TST.MMD_D import TST_MMD_D
        # H_MMD_D, _, _ = TST_MMD_D(args.name, args.N1[dd], kk+args.rs[dd], args.check, args.n_test, args.n_per, args.alpha, check_device(),  torch.float, args.xin_MMD_D, args.H_MMD_D, args.xout_MMD_D, args.ne_MMD_D, args.bs_MMD_D, args.lr_MMD_D)
        # print('MMD-D Done!')

        # # MEmabid
        from baseline_TST.MEmabid import TST_MEmabid
        H_MEmabid, _, _ = TST_MEmabid(args.name,args.N1[dd], kk+args.rs[dd], args.check, args.n_test, args.alpha, check_device(),  torch.float, args.tl_MEmabid, args.beta_MEmabid, args.ne_MEmabid, args.bs_MEmabid, args.lr_MEmabid)
        print('MEmabid Done!')

        # Save outputs into the Results array.
        Results[dd, 0, kk] = H_sel.sum()    / float(args.n_test)
        Results[dd, 1, kk] = H_no_sel.sum() / float(args.n_test)
        Results[dd, 2, kk] = H_Fuse.sum()   / float(args.n_test)
        Results[dd, 3, kk] = H_Agg.sum()    / float(args.n_test)
        Results[dd, 4, kk] = H_C2ST_L.sum()    / float(args.n_test)
        Results[dd, 5, kk] = H_MMD_D.sum() / float(args.n_test)
        Results[dd, 6, kk] = H_MEmabid.sum()   / float(args.n_test)
        
        # Save p-values from TST_MMD_AU; set p-values from other methods to 0.
        Results_P[dd, 0, kk] = P_sel.sum()    / float(args.n_test)
        Results_P[dd, 1, kk] = P_no_sel.sum() / float(args.n_test)
        Results_P[dd, 2, kk] = P_Fuse.sum()   / float(args.n_test)
        Results_P[dd, 3, kk] = P_Agg.sum()    / float(args.n_test)
        
        # Save intermediate results.
        if args.check == 1:
            np.savetxt('Results/test_power/'  + 'IR_' + args.name + '_testpower_' + str(args.is_cov_tr_MMD_AU) + str(args.N1), Results.reshape(Results.shape[0], -1),     fmt='%.3f')
            np.savetxt('Results/test_power/'  + 'IR_' + args.name + '_pvalue_'    + str(args.is_cov_tr_MMD_AU) + str(args.N1), Results_P.reshape(Results_P.shape[0], -1), fmt='%.10f')
        else:
            np.savetxt('Results/typeI_error/' + 'IR_' + args.name + '_typeI_error_' + str(args.is_cov_tr_MMD_AU) + str(args.N1), Results.reshape(Results.shape[0], -1),     fmt='%.3f')
            np.savetxt('Results/typeI_error/' + 'IR_' + args.name + '_pvalue_'      + str(args.is_cov_tr_MMD_AU) + str(args.N1), Results_P.reshape(Results_P.shape[0], -1), fmt='%.10f')

    # After finishing n_exp repeats, aggregate results for this condition.
    for ver in range(Results.shape[1]):
        Final_results[dd, ver, 0]   = Results[dd, ver, :].mean()
        Final_results[dd, ver, 1]   = Results[dd, ver, :].std() / np.sqrt(float(args.n_exp))

    for ver in range(Results_P.shape[1]):
        Final_results_P[dd, ver, 0] = Results_P[dd, ver, :].mean()
        Final_results_P[dd, ver, 1] = Results_P[dd, ver, :].std() / np.sqrt(float(args.n_exp))

    if args.check == 1:
        np.savetxt('Results/test_power/'  + args.name + '_testpower_'   + str(args.is_cov_tr_MMD_AU) + str(args.N1),   Final_results.reshape(Final_results.shape[0], -1),     fmt='%.3f')
        np.savetxt('Results/test_power/'  + args.name + '_pvalue_'      + str(args.is_cov_tr_MMD_AU) + str(args.N1),   Final_results_P.reshape(Final_results_P.shape[0], -1), fmt='%.10f')
    else:
        np.savetxt('Results/typeI_error/' + args.name + '_typeI_error_' + str(args.is_cov_tr_MMD_AU) + str(args.N1), Final_results.reshape(Final_results.shape[0], -1),       fmt='%.3f')
        np.savetxt('Results/typeI_error/' + args.name + '_pvalue_'      + str(args.is_cov_tr_MMD_AU) + str(args.N1), Final_results_P.reshape(Final_results_P.shape[0], -1),   fmt='%.10f')

    # Print the aggregated results for this condition.
    print(f"{args.name}, N1 = {args.N1[dd]}, check = {args.check}", flush=True)
    print("TST_MMD_AU with selection inference:    {:.3f} ± {:.3f}".format(Final_results[dd, 0, 0], Final_results[dd, 0, 1]), flush=True)
    print("TST_MMD_AU without selection inference: {:.3f} ± {:.3f}".format(Final_results[dd, 1, 0], Final_results[dd, 1, 1]), flush=True)
    print("TST_MMDFuse:                            {:.3f} ± {:.3f}".format(Final_results[dd, 2, 0], Final_results[dd, 2, 1]), flush=True)
    print("TST_MMDAgg:                             {:.3f} ± {:.3f}".format(Final_results[dd, 3, 0], Final_results[dd, 3, 1]), flush=True)
    print("C2ST:                                   {:.3f} ± {:.3f}".format(Final_results[dd, 4, 0], Final_results[dd, 4, 1]), flush=True)
    print("MMD-D:                                  {:.3f} ± {:.3f}".format(Final_results[dd, 5, 0], Final_results[dd, 5, 1]), flush=True)
    print("MEmabid:                                {:.3f} ± {:.3f}".format(Final_results[dd, 6, 0], Final_results[dd, 6, 1]), flush=True)

# Print the total execution time.
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} s", flush=True)