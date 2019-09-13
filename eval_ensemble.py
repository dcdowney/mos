import argparse
import os
import numpy as np

parser = argparse.ArgumentParser(description='Utility for ensembling MoS LSTMs')
parser.add_argument('--test_outfiles', type=str, default='.',
                    help='location of test_outfiles to compute the combined ppl of')

args = parser.parse_args()

all_lls = []

for fn in os.listdir(args.test_outfiles):
    lls = []
    with open(os.path.join(args.test_outfiles, fn)) as f:
        for s in f:
            lls.append(float(s))
    all_lls.append(lls)

arr = np.asarray(all_lls)
arr = np.transpose(arr)

modelLLs = arr.shape[1] # holds likelihood of each model

ensLL = 0 # holds likelihood of ensemble

tokens = len(arr)

for t in arr:
    modelLLs += t
    dist = np.exp(t)
    ens = np.average(dist)
    ensLL += np.log(ens)

print(f"output shape:{arr.shape}")
print(f"modelLLs: {modelLLs/tokens}")
print(f"ensembleLL: {ensLL/tokens}")