import nats_bench

print('NATS-Bench version: {:}'.format(nats_bench.version()))
# Create the API for tologoy search space
api = nats_bench.create(None, 'tss', fast_mode=True, verbose=True)

# Find the best architecture on CIFAR-10 validation set
api.verbose = False
best_arch_index, highest_valid_accuracy = api.find_best(dataset='cifar10-valid', metric_on_set='x-valid', hp='12')


print('{:} The best architecture on CIFAR-10 validation set with 12-epoch training is: [{:}] {:}'.format(
    nats_bench.api_utils.time_string(), best_arch_index, api.arch(best_arch_index)))