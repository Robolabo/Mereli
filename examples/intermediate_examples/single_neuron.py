import copy
import numpy as np
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True

import matplotlib.pyplot as plt
from mereli.globals import global_states
from mereli.neural_networks import NeuralNetwork

dt = 0.01

tau_vals = [10,20, 30, 40, 50, 100, 200]
bias_vals = [-3,-2,-1, 0, 1, 2, 3]
gain_vals = [0.5, .1, 0.2, 0.5, 1, 2,  5]
w_vals = [-3, -2,-1, 0, 1,2, 3]



var = 'g'
varY = 'voltage'
# varY = 'output'
values = {'g' : gain_vals, 'tau' : tau_vals, 'beta' : bias_vals, 'mathrm{I}' :  w_vals}
ann_dict = {}

for val in values[var]:
    g = 1
    b = 0 
    w = 3
    tau_i=20
    if var == 'g': 
        g = val
    elif var == 'tau':
        tau_i = val
    elif var == '\beta':
        b = val 
    else: 
        w = val
    ann = NeuralNetwork(dt, neuron_model='rate_model', synapse_model='static_synapse')
    ann.add_stimuli('I', 1)
    ann.add_ensemble('Motor', 1, tau=tau_i*dt, bias=b, gain=g)
    ann.set_motor('Motor')
    ann.add_synapse('I-M', 'I', 'Motor', weight=w)
    ann.add_decoder('IdentityDecoding', 'Motor', 'Action1')
    ann.build()
    ann.reset()

    ann_dict[f'{var}={val}'] = copy.deepcopy(ann)

stimuli =  np.ones(200)

out_dict= {k : [] for k in ann_dict.keys()}
for k, ann_i in ann_dict.items():
    outputs = []
    voltages = [] 
    ann_i.reset()
    # if 'beta' not in k:
    #     # outputs.append([0.5]) 
    #     outputs.append([0.5]) 
    # voltages.append(ann_i.voltage_of('Motor', 0))
    for in1 in stimuli: 
        output = ann_i.step({'I' : in1})['Action1']
        outputs.append(output)
        voltages.append(ann_i.voltage_of('Motor', 0))
    if varY == 'voltage': 
        out_dict[k] = np.array(voltages).flatten()
    else:
        out_dict[k] = np.array(outputs).flatten()


T = len(outputs)
plt.figure(figsize=(7.5,7.5))

# plt.plot(np.arange(T) * dt, input1, color='k', lw=3)
# plt.plot(np.arange(T) * dt, input2, color='r', lw=3)

# plt.plot(np.arange(T) * dt, voltages['m'], color='k', lw=2.5)
for output in out_dict.values():
    plt.plot(np.arange(T) * dt, output,  lw=4)

if varY == 'voltage':
    plt.ylabel(r'$v(t)$', fontsize=35)
else: 
    plt.ylabel(r'$u(t)$', fontsize=35)
plt.xlabel(r'$t (s)$', fontsize=35)

legend_loc = {'tau' : 'lower right'}.get(var, 'upper right')
if var in ['tau', 'beta', 'mathrm{I}']: 
    plt.legend([r'$\{}={}$'.format(var, str(v)) for v in values[var]], fontsize=25, loc=legend_loc,  framealpha=1) 
elif var == 'I':
    plt.legend([r'${}={}$'.format(var, str(v)) for v in values[var]], fontsize=25, loc=legend_loc,  framealpha=1) 
else:
    plt.legend([r'${}={}$'.format(var, str(v)) for v in values[var]], fontsize=25, loc=legend_loc,  framealpha=1) 
plt.locator_params(axis='x', nbins=5)
plt.xlim([0, T*dt])
if varY == 'output':
    plt.ylim([0, 1.1])
else: 
    plt.ylim([-3.3, 3.3])
plt.yticks(fontsize=30)
plt.xticks(fontsize=30)
# plt.title('Input Layer', fontsize=30)
aa = np.random.randint(1)
plt.rcParams['figure.constrained_layout.use'] = True
plt.subplots_adjust(left=0.2, right=0.95, top=0.98, bottom=0.2)


filename = f'/Users/rsendra/mereli_scripts/neuron_{var}_{varY}.png'
plt.savefig(filename, dpi=800)
plt.show()

# axs[1].plot(np.arange(T) * dt, voltages['h1'], color='k')
# axs[1].plot(np.arange(T) * dt, voltages['h2'], color='r')
# axs[1].set_ylabel('Voltage')
# axs[1].set_xlabel('Time')
# axs[1].set_title('Hidden Layer')
# axs[1].legend(['v1', 'v2'], loc="upper right")


# axs[2].plot(np.arange(T) * dt, voltages['m'], color='k')
# axs[2].set_ylabel('Voltage')
# axs[2].set_xlabel('Time')
# axs[2].set_title('Output Layer')

# plt.xlabel('Time (s)')
# plt.ylabel('Measure')
# plt.legend(['Stimuli', 'Output/Action', 'Voltage/Neuron State'])
# plt.show()



