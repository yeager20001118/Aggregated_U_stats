# MMD_AU        MMDFuse       MMDAgg        MMD-D         AutoTST       MEmabid  
# 0.134 ± 0.016 0.153 ± 0.055 0.124 ± 0.055 0.081 ± 0.013 0.124 ± 0.049 0.128 ± 0.012
# 0.454 ± 0.022 0.346 ± 0.065 0.270 ± 0.076 0.172 ± 0.057 0.254 ± 0.049 0.136 ± 0.032
# 0.750 ± 0.028 0.673 ± 0.084 0.539 ± 0.073 0.187 ± 0.052 0.363 ± 0.098 0.314 ± 0.062
# 0.915 ± 0.016 0.863 ± 0.042 0.764 ± 0.053 0.340 ± 0.084 0.722 ± 0.097 0.511 ± 0.071
# 0.983 ± 0.005 0.972 ± 0.011 0.941 ± 0.026 0.494 ± 0.099 0.809 ± 0.089 0.608 ± 0.075
# 0.998 ± 0.001 0.998 ± 0.002 0.987 ± 0.008 0.645 ± 0.087 0.932 ± 0.026 0.810 ± 0.052

# '--is_cov_tr_MMD_AU' = False '--is_cov_MMD_AU' = True
# 0.134 ± 0.016 0.116 ± 0.015 0.153 ± 0.055 0.000 ± 0.000
# 0.454 ± 0.022 0.336 ± 0.025 0.346 ± 0.065 0.000 ± 0.000
# 0.750 ± 0.028 0.619 ± 0.028 0.673 ± 0.084 0.000 ± 0.000
# 0.915 ± 0.016 0.828 ± 0.017 0.863 ± 0.042 0.000 ± 0.000
# 0.983 ± 0.005 0.948 ± 0.012 0.972 ± 0.011 0.000 ± 0.000
# 0.998 ± 0.001 0.988 ± 0.003 0.998 ± 0.002 0.000 ± 0.000

# 0.134 ± 0.016 0.116 ± 0.015 seed 230
# 0.192 ± 0.020 0.145 ± 0.018 seed 230

# 0.983 ± 0.005 0.948 ± 0.012 seed 580
# 0.958 ± 0.009 0.909 ± 0.012  seed 485

# 0.998 ± 0.001 0.988 ± 0.003 seed 680
# 0.998 ± 0.001 0.989 ± 0.003 seed 580


# '--is_cov_tr_MMD_AU' = False '--is_cov_MMD_AU' = False
# 0.061 ± 0.007 0.059 ± 0.007 0.000 0.000 0.000 0.000
# 0.269 ± 0.024 0.226 ± 0.020 0.000 0.000 0.000 0.000
# 0.406 ± 0.021 0.357 ± 0.020 0.000 0.000 0.000 0.000
# 0.565 ± 0.019 0.518 ± 0.018 0.000 0.000 0.000 0.000
# 0.662 ± 0.029 0.615 ± 0.030 0.000 0.000 0.000 0.000
# 0.793 ± 0.022 0.751 ± 0.024 0.000 0.000 0.000 0.000

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
parser.add_argument('--name',               default='BLOB',                 help='Dataset name')
# Accept multiple sample sizes (one for each experimental condition)
parser.add_argument('--check',              default=1,                       help='Indicator: 1 for test power, 0 for Type-I error')
parser.add_argument('--n_exp',              default=10,                      help='Number of experiment repeats')
parser.add_argument('--n_test',             default=100,                     help='Number of two-sample test executions')
parser.add_argument('--n_per',              default=100,                     help='Number of permutations for bootstrap replications')
parser.add_argument('--alpha',              default=0.05,                    help='Significance level for tests')

# parser.add_argument('--rs',                 default=[230,                300,                380,                484,                580,                680],          help='Base random seed')
parser.add_argument('--rs',                 default=[230,                300,                380,                484,                485,                580],              help='Base random seed')
parser.add_argument('--N1',                 default=[50,                 100,                150,                200,                250,                300],           help='Sample sizes for each experimental condition')
parser.add_argument('--is_cov_tr_MMD_AU',   default=False,                          help='Flag to use training covariance in the testing procedure of MMD_AU (True/False)')
parser.add_argument('--way',                default=['Agg', 'Agg', 'Agg', 'Agg', 'Agg', 'Agg'],    help="Bandwidth search strategy for each kernel type (e.g. 'Grid', 'Boost')")


# Parameters specific to the MMD_AU method in training phase.
parser.add_argument('--lr_MMD_AU',          default=[0.0005,             0.0005,             0.0005,             0.0005,             0.0005,             0.0005],      help='Learning rate for MMD_AU optimization')
# Bandwidth parameters: expects four sub-lists (one per experimental condition) each with four integers.
parser.add_argument('--n_bandwidth_MMD_AU', default=[[1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0], [1, 0, 1, 0, 0, 0]], help='Bandwidth settings for kernels as list of lists: (gaussian, laplacian, deep_gaussian, deep_laplacian)')
parser.add_argument('--ne_MMD_AU',          default=[1000,               1000,               1000,               1000,               1000,               1000],          help='Number of training epochs for MMD_AU')
parser.add_argument('--bs_MMD_AU',          default=[100,                100,                100,                100,                100,                100],           help='Batch size for MMD_AU training')
parser.add_argument('--is_cov_MMD_AU',      default=[True,               True,               True,               True,               True,               True],         help='Flag to use covariance in MMD_AU (True/False)')

args = parser.parse_args()

# Determine the number of experimental conditions.
n_conditions = len(args.N1)
# For MMD_AU, record two versions: with and without selection inference.
# However, since you also call TST_MMDFuse and TST_MMDAgg, we'll store outputs for 4 versions.
n_versions = 4
# Initialize result arrays.
Results   = np.zeros((n_conditions, n_versions, int(args.n_exp)))
Results_P = np.zeros_like(Results)
# Arrays to store aggregated (final) results: [mean, standard error].
Final_results   = np.zeros((n_conditions, n_versions, 2))
Final_results_P = np.zeros_like(Final_results)

# # Set up an encoder based on TorchVision's ResNet18 for MMD_AU.
import torch.nn as nn
np.random.seed(1102)
torch.manual_seed(1102)
torch.cuda.manual_seed_all(1102)  # 如果有GPU
class ModelLatentF(nn.Module):
    """define deep networks."""
    def __init__(self, x_in=2, H=50, x_out=50):
        """Init latent features."""
        super(ModelLatentF, self).__init__()
        self.restored = False

        self.latent = nn.Sequential(
            nn.Linear(x_in, H, bias=True),
            nn.Softplus(),
            nn.Linear(H, H, bias=True),
            nn.Softplus(),
            nn.Linear(H, H, bias=True),
            nn.Softplus(),
            nn.Linear(H, x_out, bias=True),
        )
    def forward(self, input):
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant

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
    #     encoder = ModelLatentF().to(check_device())
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
        # with open("model/"  + args.name + "_" + str(args.n_bandwidth_MMD_AU[dd]) + "_"  + str(args.is_cov_MMD_AU[dd]) + "_"  + str(args.N1[dd]) + "_"  + str(args.rs[dd])  + "_"  + str(args.ne_MMD_AU[dd]) + "_"  + str(args.bs_MMD_AU[dd]) + "_"  + str(args.lr_MMD_AU[dd])  + ".pkl", "wb") as f:
        #     pickle.dump(model_au, f)

    for kk in range(int(args.n_exp)):
        # Preallocate arrays for each method's raw output for the current condition.
        H_sel    = np.zeros(args.n_test)  # For rejection results of TST_MMD_AU with selection inference.
        H_no_sel = np.zeros(args.n_test)  # For rejection results of TST_MMD_AU without selection inference.
        H_Fuse   = np.zeros(args.n_test)  # For rejection results of TST_MMDFuse.
        H_Agg    = np.zeros(args.n_test)  # For rejection results of TST_MMDAgg.
        
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
        # ----- TST_MMDAgg Method -----
        from baseline_TST.MMDAgg import TST_MMDAgg
        H_Agg, P_Agg, training_time_Agg, testing_time_Agg = TST_MMDAgg(
            args.name,
            args.N1[dd],
            kk + args.rs[dd],
            args.check,
            args.n_test,
            5*args.n_per,
            args.alpha
        )
        
        # Save outputs into the Results array.
        Results[dd, 0, kk] = H_sel.sum()    / float(args.n_test)
        Results[dd, 1, kk] = H_no_sel.sum() / float(args.n_test)
        Results[dd, 2, kk] = H_Fuse.sum()   / float(args.n_test)
        Results[dd, 3, kk] = H_Agg.sum()    / float(args.n_test)
        
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

# Print the total execution time.
end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time} s", flush=True)