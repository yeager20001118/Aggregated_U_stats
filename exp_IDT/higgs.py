# check = 1 for test power check = 0 for Type-I error
import numpy as np
import torch
import argparse
parser = argparse.ArgumentParser()
import sys
import os
sys.path.append(os.path.abspath('/data/gpfs/projects/punim2335/baselines/'))

DEFAULT_N=1000

# parameters to generate data
parser.add_argument('--name',  default='HIGGS', help = 'Dataset')
parser.add_argument('--N1',    default=[100, 300, 500, 800, 1000, 1200],    help = 'Size of each sample')
parser.add_argument('--check', default=1,      help = '1 for test power; 0 for type-I error')
parser.add_argument('--rs',    default=[170, 5135, 351, 353, 1113, 274],      help = 'Random seed')

# parameters of experimental setting
parser.add_argument('--n_exp',  default=10,                   help='Number of experiment runs')
parser.add_argument('--n_test', default=100,                   help='Number of independence test runs')
parser.add_argument('--n_per',  default=100,                   help='Number of permutation test runs')
parser.add_argument('--alpha',  default=0.05,                  help='Confidence level of independence test')
parser.add_argument('--device', default=torch.device("cpu"),  help='Device of data')
parser.add_argument('--dtype',  default=torch.float,           help='Dtype of data')

# parameters of HSICAgg
parser.add_argument('--R', default=[99, 299, 499, 799, 999, 1199], help= 'Number of superdiagonals to consider')

# parameters of HSIC-AU
parser.add_argument('--N_epoch', default=[50, 20, 50, 50, 100, 100], help='Number of epochs')
parser.add_argument('--batch_size', default=128, help='Batch size')
parser.add_argument('--lr', default=0.0005, help='Learning rate')
parser.add_argument('--n_bandwidth', default=[(4, 4, 5, 0, 0, 0), (2, 2, 5, 0, 0, 0), (4, 4, 5, 0, 0, 0), (4, 4, 5, 0, 0, 0), (2, 2, 5, 0, 0, 0), (2, 2, 5, 0, 0, 0)], help='Number of bandwidths')
parser.add_argument('--reg', default=1e-8, help='Regularization parameter')
parser.add_argument('--way', default=['Fuse', 'Fuse', 'Agg', 'Agg', 'Agg', 'Agg'], help='Way of HSIC-AU')
parser.add_argument('--is_cov', default=True, help='Whether to use covariance matrix')
parser.add_argument('--is_cov_tr', default=True, help='Whether to use covariance matrix for training')

args = parser.parse_args()

Results = np.zeros((9, args.n_exp)) # vary the number depending on # baselines

# AU_S,        AU,         Agg200,    FSIC, HSIC_O, Agg1, AggCom, HSIC_W, HSIC
# 0.136±0.014 0.107±0.012 0.103±0.026                                           N100 seed858 bw445000 ep0
# 0.397±0.022 0.261±0.019 0.376±0.058                                           N300 seed5131 bw445000 ep0
# 0.762±0.015 0.587±0.015 0.715±0.066                                           N500 seed1162 bw445000 ep0

# 0.104±0.009 0.086±0.008 0.079±0.013                                           N100 seed170 bw445000 ep50
# 0.416±0.022 0.306±0.018 0.407±0.059                                           N300 seed5135 bw225000 ep20
# 0.635±0.023 0.540±0.023 0.617±0.056                                           N500 seed351 bw445000 ep50
# 0.850±0.016 0.812±0.022 0.792±0.050                                           N800 seed353 bw445000 ep50
# 0.959±0.005 0.930±0.008 0.938±0.027                                           N1000 seed1113 bw225000 ep50
# 0.995±0.002 0.984±0.006 0.966±0.010                                           N1200 seed274 bw225000 ep100

H_HSICAgg1 = np.zeros(args.n_test)
H_HSICAgg200 = np.zeros(args.n_test)
H_HSICAggCom = np.zeros(args.n_test)
H_FSIC = np.zeros(args.n_test)
H_HSIC_O = np.zeros(args.n_test)
H_HSIC_W = np.zeros(args.n_test)
H_HSIC = np.zeros(args.n_test)
H_HSIC_AU_select = np.zeros(args.n_test)
H_HSIC_AU_no_select = np.zeros(args.n_test)

# from baseline_IDT.HSIC_AU import IDT_HSIC_AU, train_HSIC_AU
from baseline_IDT.HSICAggInc import IDT_HSICAgg
for i in range(len(args.N1)):
    if i != 4:
        continue
    N1 = args.N1[i]
    rs = args.rs[i]
    # n_bandwidth = args.n_bandwidth[i]
    # model_au = train_HSIC_AU(args.name, N1, rs, args.check, args.N_epoch[i], args.batch_size, args.lr, n_bandwidth, args.reg, args.way, args.is_cov, encoder=None)
    for kk in range(args.n_exp): 
        # H_HSICAgg1, _, _ = IDT_HSICAgg(args.name, N1, kk+rs, args.check, args.n_test, args.alpha, R=100)
        # print(f'HSICAgg R={1} Done!')
        
        # H_HSICAgg200, _, _ = IDT_HSICAgg(args.name, N1, kk+rs, args.check, args.n_test, args.alpha, R=args.R[i])#R=200)
        # print(f'HSICAgg R={200} Done!')
        
        # H_HSICAggCom, _, _ = IDT_HSICAgg(args.name, args.N1, kk+args.rs, args.check, args.n_test, args.alpha, args.R)
        # print(f'HSICAgg R={args.R} Done!')
        
        # from baseline_IDT.FSIC import IDT_FSIC
        # H_FSIC, _, _ = IDT_FSIC(args.name, N1, kk+rs, args.check, args.n_test, args.alpha)
        # print(f'FSIC Done!')
        
        # from baseline_IDT.HSIC_O import IDT_HSIC_O
        # H_HSIC_O, _, _ = IDT_HSIC_O(args.name, N1, kk+rs, args.check, args.n_test, args.alpha)
        # print(f'HSIC-O Done!')
        
        # from baseline_IDT.HSIC_W import IDT_HSIC_W
        # H_HSIC_W, _, _ = IDT_HSIC_W(args.name, args.N1, kk+args.rs, args.check, args.n_test, args.alpha)
        # print(f'HSIC-W Done!')
        
        # from baseline_IDT.HSIC import IDT_HSIC
        # H_HSIC, _, _ = IDT_HSIC(args.name, N1, kk+rs, args.check, args.n_test, args.alpha)
        # print(f'HSIC Done!')

        H_HSIC_AU_select, H_HSIC_AU_no_select, _, _, _, _ = IDT_HSIC_AU(args.name, N1, kk+rs, args.check, args.n_test, args.n_per, args.alpha, args.is_cov_tr, model_au)
        # print(f'HSIC-AU Done!')
        
        Results[0, kk] = H_HSICAgg1.sum() / args.n_test
        Results[1, kk] = H_HSICAgg200.sum() / args.n_test
        Results[2, kk] = H_HSICAggCom.sum() / args.n_test
        Results[3, kk] = H_FSIC.sum() / args.n_test
        Results[4, kk] = H_HSIC_O.sum() / args.n_test
        Results[5, kk] = H_HSIC_W.sum() / args.n_test
        Results[6, kk] = H_HSIC.sum() / args.n_test
        Results[7, kk] = H_HSIC_AU_select.sum() / args.n_test
        Results[8, kk] = H_HSIC_AU_no_select.sum() / args.n_test

        if args.check == 1:
            os.makedirs("./Results/test_power/", exist_ok=True)
            np.savetxt('./Results/test_power/'+args.name+'_Results_'+str(N1)+'_'+str(args.n_exp), Results, fmt='%.3f')
        else:
            os.makedirs("./Results/typeI_error/", exist_ok=True)
            np.savetxt('./Results/typeI_error/'+args.name+'_Results_'+str(N1)+'_'+str(args.n_exp), Results, fmt='%.3f')
    
    Final_results = np.zeros((Results.shape[0],2))

    for i in range(Results.shape[0]):
        Final_results[i][0] = Results[i].sum()/args.n_exp
        Final_results[i][1] = Results[i].std()/np.sqrt(args.n_exp)

    if args.check == 1:
        np.savetxt('./Results/test_power/'+args.name+'_'+str(N1)+'_'+str(args.n_exp), Final_results, fmt='%.3f')
    else:
        np.savetxt('./Results/typeI_error/'+args.name+'_'+str(N1)+'_'+str(args.n_exp), Final_results, fmt='%.3f')

    if args.check == 1:
        print(args.name, ", N1 = ", str(N1), ", test power of ",  str(args.n_exp), " experiment runs")
    else:
        print(args.name, ", N1 = ", str(N1), ", type-I error of ",  str(args.n_exp), " experiment runs")

    print("HSICAgg R=100: {:.3f}±{:.3f}".format(Results[0].sum()/args.n_exp, Results[0].std()/np.sqrt(args.n_exp)))
    # print("HSICAgg R=200: {:.3f}±{:.3f}".format(Results[1].sum()/args.n_exp, Results[1].std()/np.sqrt(args.n_exp)))
    # print("HSICAggCom R={:.3f}: {:.3f}±{:.3f}".format(args.R, Results[2].sum()/args.n_exp, Results[2].std()/np.sqrt(args.n_exp)))
    # print("FSIC: {:.3f}±{:.3f}".format(Results[3].sum()/args.n_exp, Results[3].std()/np.sqrt(args.n_exp)))
    # print("HSIC-O: {:.3f}±{:.3f}".format(Results[4].sum()/args.n_exp, Results[4].std()/np.sqrt(args.n_exp)))
    # print("HSIC-W: {:.3f}±{:.3f}".format(Results[5].sum()/args.n_exp, Results[5].std()/np.sqrt(args.n_exp)))
    # print("HSIC: {:.3f}±{:.3f}".format(Results[6].sum()/args.n_exp, Results[6].std()/np.sqrt(args.n_exp)))
    # print("HSIC-AU select: {:.3f}±{:.3f}".format(Results[7].sum()/args.n_exp, Results[7].std()/np.sqrt(args.n_exp)))
    # print("HSIC-AU no select: {:.3f}±{:.3f}".format(Results[8].sum()/args.n_exp, Results[8].std()/np.sqrt(args.n_exp)))