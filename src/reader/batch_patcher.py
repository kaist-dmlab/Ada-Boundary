import numpy as np
import time, os, math, operator, statistics, sys
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from random import Random

class Sample(object):
    def __init__(self, id, image, label):
        self.id = id
        self.image = image
        self.label = label

class MiniBatch(object):
    def __init__(self):
        self.ids = []
        self.images = []
        self.labels = []

    def append(self, id, image, label):
        self.ids.append(id)
        self.images.append(image)
        self.labels.append(label)

    def get_size(self):
        return len(self.ids)

class Quantizer(object):
    def __init__(self, size_of_data, min, max):

        # size of data must be even number
        self.size_of_data = size_of_data
        self.half_size_of_data = int(size_of_data/2)
        self.max = max
        self.min = min
        self.step_size = self.max / float(size_of_data)
        self.doubled_step_size = 2.0*self.step_size
        self.quantization_indexes = {}

    def quantizer_func_for_boudnary(self, distance):
        if distance >= 0:
            # Positive sample
            index = int(math.ceil(distance / self.step_size))
        else:
            # Negative sample
            index = -int(math.floor(distance / self.step_size))
        return index

    def quantizer_func_for_easy(self, distance):
        if distance >= 0:
            # Positive sample
            index = -int(math.ceil(distance / self.doubled_step_size)) + self.half_size_of_data + 1
        else:
            # Negative sample
            index = -int(math.floor(distance / self.doubled_step_size)) + self.half_size_of_data
        return index

    def quantizer_func_for_hard(self, distance):
        if distance >= 0:
            # Positive sample
            index = int(math.ceil(distance / self.doubled_step_size)) + self.half_size_of_data
        else:
            # Negative sample
            index = int(math.floor(distance / self.doubled_step_size)) + self.half_size_of_data + 1
        return index

# For computing emphirical distribution F(x), we adopt binning approach based on bins
class Binning(object):
    def __init__(self, size_of_data, min, max, num_of_bins):
        self.size_of_data = size_of_data
        self.num_of_bins = num_of_bins
        self.max = max
        self.min = min
        self.step_size = (self.max-self.min) / float(self.num_of_bins)
        self.bins = {}

        if num_of_bins % 2 != 0:
            print("num_of_bins must be even value.")
        self.half_num_bins = int(self.num_of_bins/2)

        # For only Ada-Uniform method
        # Inverted bin_id index for fast asynch update
        self.inverted_index = np.zeros(self.size_of_data, dtype=float)

        # Collect possible bin ids
        self.bin_ids = []
        for i in range(1, self.half_num_bins+1):
            self.bins[i] = []
            self.bin_ids.append(i)
            self.bins[-i] = []
            self.bin_ids.append(-i)

        # Random bin initialization
        rand_bin_ids = np.random.choice(self.bin_ids, self.size_of_data)
        for i in range(len(rand_bin_ids)):
            self.bins[rand_bin_ids[i]].append(i)
            self.inverted_index[i] = rand_bin_ids[i]

    def get_bin_id(self, distance):
        if distance >= 0:
            # Positive sample
            bin_id = int(math.ceil(distance / self.step_size))
        else:
            # Negative sample
            bin_id = int(math.floor(distance / self.step_size))

        if bin_id > self.half_num_bins:
            bin_id = self.half_num_bins
        elif bin_id < -self.half_num_bins:
            bin_id = -self.half_num_bins

        return bin_id

    def asynch_update_bins(self, ids, distances):
        # Update only partial information
        for i in range(len(ids)):
            prev_bin_id = self.inverted_index[ids[i]]
            # Remove previous info
            self.bins[prev_bin_id].remove(ids[i])
            # Update current info
            cur_bin_id = self.get_bin_id(distances[i])
            self.bins[cur_bin_id].append(ids[i])
            self.inverted_index[ids[i]] = cur_bin_id


# update_method in [random, boundary, hard, easy, distance]
class ProbTable(object):
    def __init__(self, size_of_data, num_of_classes, s_e, update_method):
        self.size_of_data = size_of_data
        self.num_of_classes = num_of_classes
        self.s_e = s_e
        self.update_method = update_method
        self.max_distance = np.sqrt((float(num_of_classes) - 1.0) / (float(num_of_classes) * float(num_of_classes)))
        self.min_distance = - self.max_distance
        self.fixed_term = math.exp(math.log(s_e) / size_of_data)
        self.table = np.ones(self.size_of_data, dtype=float)

        self.quantizer = Quantizer(self.size_of_data, self.min_distance, self.max_distance)
        if update_method == "boundary":
            self.quantizer_func = self.quantizer.quantizer_func_for_boudnary
        elif update_method == "easy":
            self.quantizer_func = self.quantizer.quantizer_func_for_easy
        elif update_method == "hard":
            self.quantizer_func = self.quantizer.quantizer_func_for_hard
        elif update_method == "uniform":
            self.binning = Binning(self.size_of_data, self.min_distance, self.max_distance, num_of_bins=40)

        # For Ada-Boundary/Easy/Hard
        # Initialize table : Set all importance to max importance, then all samples are chosen properly at 1 iteration
        for i in range(self.size_of_data):
            self.table[i] = math.pow(self.fixed_term, 1)

    ################################ Update Method ####################################

    # For Ada-Boundary/Easy/Hard methods
    def get_sampling_probability(self, quantization_index):
        return 1.0 / math.pow(self.fixed_term, quantization_index)

    def async_update_prob_table(self, ids, distances):
        for i in range(len(ids)):
            self.table[ids[i]] = self.get_sampling_probability(self.quantizer_func(distances[i]))

    # For Ada-Uniform: in this cast, a few bin changes give global effect of importance table
    def bulk_update_prob_table(self, ids, distances):
        # Update bins at first
        self.binning.asynch_update_bins(ids, distances)

        # Calculate emphirical distibution & use it's inverse the value as the sample importance
        for value in self.binning.bins.values():
            if len(value) == 0:
                importance = 0.0
            else:
                # Assign F^{-1}(x)
                importance = float(self.size_of_data) / float(len(value))
            for id in value:
                self.table[id] = importance
    ###################################################################################


# Batch Patcher
class BatchPatcher(object):
    def __init__(self, size_of_data, batch_size, num_of_classes, s_e=100.0, update_method="random"):
        # meta info
        self.size_of_data = size_of_data
        self.batch_size = batch_size
        self.num_of_classes = num_of_classes
        self.update_method = update_method
        self.num_iters_per_epoch = int(math.ceil(float(size_of_data) / float(batch_size)))

        # importance table
        self.prob_table = ProbTable(size_of_data, num_of_classes, s_e, self.update_method)

        # For in-memory mini-batch generation
        self.loaded_data = []

        # Replacement in mini-batch for random batch selection
        self.replacement = True

    def bulk_load_in_memory(self, sess, ids, images, labels):
        # initialization
        self.loaded_data = []
        for i in range(self.size_of_data):
            self.loaded_data.append(None)

        # load data set in memory
        set_test = set()

        # while len(self.loaded_data) < self.size_of_data:
        for i in range(self.num_iters_per_epoch * 2):
            mini_ids, mini_images, mini_labels = sess.run([ids, images, labels])

            for j in range(self.batch_size):
                id = bytes_to_int(mini_ids[j])
                if not id in set_test:
                    self.loaded_data[id] = Sample(bytes_to_int(mini_ids[j]), mini_images[j], bytes_to_int(mini_labels[j]))
                    set_test.add(bytes_to_int(mini_ids[j]))

        print("# of disjoint samples: ", len(self.loaded_data))

    def update_prob_table(self, ids, distances):
        if self.update_method == "uniform":
            self.prob_table.bulk_update_prob_table(ids, distances)
        else:
            self.prob_table.async_update_prob_table(ids, distances)

    def get_next_mini_batch(self, num_of_sample, is_warm_up=True, p_table = None):

        if self.update_method == "random" or is_warm_up:
            selected_sample_ids = np.random.choice(self.size_of_data, num_of_sample, self.replacement)
        else:
            if p_table is None:
                total_sum = np.sum(self.prob_table.table)
                p_table = self.prob_table.table / total_sum
            selected_sample_ids = np.random.choice(self.size_of_data, num_of_sample, self.replacement, p=p_table)

        # Fetch mini-batch samples from loaded_data (in memory)
        mini_batch = MiniBatch()
        for id in selected_sample_ids:
            sample = self.loaded_data[id]
            mini_batch.append(sample.id, sample.image, sample.label)

        return mini_batch.ids, mini_batch.images, mini_batch.labels

    def get_init_mini_batch(self, init_id):
        # init_id from 0~self.num_iters_per_epoch
        selected_sample_ids = list(range(init_id*self.batch_size, init_id*self.batch_size+self.batch_size))

        # Fetch mini-batch samples from loaded_data (in memory)
        mini_batch = MiniBatch()
        for id in selected_sample_ids:
            if id >= self.size_of_data:
                sample = self.loaded_data[0]
                mini_batch.append(sample.id, sample.image, sample.label)
            else:
                sample = self.loaded_data[id]
                mini_batch.append(sample.id, sample.image, sample.label)

        return mini_batch.ids, mini_batch.images, mini_batch.labels

    def get_normalized_table(self):
        total_sum = np.sum(self.prob_table.table)
        return self.prob_table.table / total_sum

def bytes_to_int(bytes_array):
    result = 0
    for b in bytes_array:
        result = result * 256 + int(b)
    return result
