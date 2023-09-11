import numpy as np

def get_noised_input(input_x, sigma):
    new_input = input_x + np.random.normal(0, sigma, input_x.shape)
    new_input = new_input.astype(np.float32)
    return new_input

# 修改此处，改为添加高斯噪声
def get_noised_cfs(input_x, cfs, sigmas, evaluator, generate_cf_func):

    cfs_list = []

    for idx, sigma in enumerate(sigmas):
        tmp_result = {}
        # mapsolver1 = MapSolver(n)
        new_input = get_noised_input(input_x, sigma)
        # cfsolver1 = CFSolver(n, model, new_input, to_replace, desired_pred=1)
        # _cfs = []
        # for text, cf, mask in FindCF(cfsolver1, mapsolver1):
        #     _cfs.extend(cf)
        #
        # _cfs = np.reshape(_cfs, (-1, input_x.shape[1]))

        _cfs = generate_cf_func(idx, new_input)
        diversity = evaluator.diversity(_cfs)

        tmp_result['sigma'] = np.round(sigma, 4)
        tmp_result['cf'] = cfs
        tmp_result['cf2'] = _cfs
        tmp_result['num_of_cf2'] = _cfs.shape[0]
        tmp_result['diversity'] = diversity
        cfs_list.append(tmp_result)

    return cfs_list

