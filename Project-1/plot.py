import matplotlib.pyplot as plt
from ibm import IBM
import os
import cPickle

directory = 'results/ibm2/aers'
plots_dir = 'plots'
label = 'aer'
out_filename = "ibm2-aers"

# legend_labels = [0.001, 0.01, 0.1, 0.0005, 0.005, 0.05]
legend_labels = ['uniform', 'random 1', 'random 2', 'random 3', 'IBM1 pretrained']
i = 0

max_iters = 0
filenames = sorted(os.listdir(directory))
for filename in os.listdir(directory):
    f = open(directory + '\\' + filename, 'r')
    data =  cPickle.load(f)
    print filename
    print data
    if len(data) > max_iters:
        max_iters = len(data)
    IBM.plot(data, label, legend_labels[i], plots_dir +'\\' + filename, False, max_iters)
    i = i + 1

plt.legend(bbox_to_anchor=(0.6, 0.95), loc=2, borderaxespad=0.)
plt.savefig(plots_dir +'\\' + out_filename + '.png', bbox_inches='tight')