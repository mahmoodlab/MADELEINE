import pickle
import os

def save_pkl(filename, save_object):
	writer = open(filename,'wb')
	pickle.dump(save_object, writer, protocol=pickle.HIGHEST_PROTOCOL)
	writer.close()


def load_pkl(filename):
	loader = open(filename,'rb')
	file = pickle.load(loader)
	loader.close()
	return file

def print_network(net, results_dir=None):
    num_params = 0
    num_params_train = 0

    for param in net.parameters():
        n = param.numel()
        num_params += n
        if param.requires_grad:
            num_params_train += n
    
    # print('Total number of parameters: %d' % num_params)
    # print('Total number of trainable parameters: %d' % num_params_train)

    if results_dir is not None:
        fname = "model_config.txt"
        path = os.path.join(results_dir, fname)
        f = open(path, "w")
        f.write(str(net))
        f.write("\n")
        f.write('Total number of parameters: %d \n' % num_params)
        f.write('Total number of trainable parameters: %d \n' %
                num_params_train)
        f.close()

    # print(net)

