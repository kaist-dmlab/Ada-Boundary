import numpy as np
import math, operator
import random

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

class ImportanceTable(object):
    def __init__(self, size_of_data, num_of_classes, s_e, update_method):
        self.size_of_data = size_of_data
        self.num_of_classes = num_of_classes
        self.s_e = s_e
        self.update_method = update_method
        self.fixed_term = math.exp(math.log(s_e) / size_of_data)
        self.table = np.ones(self.size_of_data, dtype=float)

        # Initialize table : Set all importance to max importance, then all samples are choiced properly at 1 iteration
        for i in range(self.size_of_data):
            self.table[i] = math.pow(self.fixed_term, 1)

    # For online batch (not our method)
    def bulk_update_importance_table_based_on_online_bs(self, loss_map):

        # Sort loss map by descending order
        loss_map = dict(sorted(loss_map.items(), key=operator.itemgetter(1), reverse=True))

        # Calculate importance based on the order
        cur_order = 1
        for key in loss_map.keys():
            self.table[key] = 1.0 / math.pow(self.fixed_term, cur_order)
            cur_order += 1


# update_method in [random, boundary, hard, easy, distance]
class BatchPatcher(object):
    def __init__(self, size_of_data, batch_size, num_of_classes, s_e=100.0, update_method="random"):
        # meta info
        self.size_of_data = size_of_data
        self.batch_size = batch_size
        self.num_of_classes = num_of_classes
        self.update_method = update_method
        self.num_iters_per_epoch = int(math.ceil(float(size_of_data) / float(batch_size)))

        # importance table
        self.importance_table = ImportanceTable(size_of_data, num_of_classes, s_e, self.update_method)

        # For in-memory mini-batch generation
        self.loaded_data = []
        self.shuffled_data = []

        # Replacement in mini-batch
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
                    self.loaded_data[id] = Sample(bytes_to_int(mini_ids[j]), mini_images[j],
                                                  bytes_to_int(mini_labels[j]))
                    set_test.add(bytes_to_int(mini_ids[j]))

        for data in self.loaded_data:
            self.shuffled_data.append(data)

        print("# of disjoint samples: ", len(self.loaded_data))

    def get_next_mini_batch(self, num_of_sample, is_warm_up=True):
        if self.update_method == "random" or is_warm_up:
            selected_sample_ids = np.random.choice(self.size_of_data, num_of_sample, self.replacement)
        else:
            total_sum = np.sum(self.importance_table.table)
            pTable = self.importance_table.table / total_sum
            selected_sample_ids = np.random.choice(self.size_of_data, num_of_sample, self.replacement, p=pTable)

        # Fetch mini-batch samples from loaded_data (in memory)
        mini_batch = MiniBatch()
        for id in selected_sample_ids:
            sample = self.loaded_data[id]
            mini_batch.append(sample.id, sample.image, sample.label)

        return mini_batch.ids, mini_batch.images, mini_batch.labels

    # For online batch method
    def update_importance_table_based_on_online_bs(self, loss_map):
        self.importance_table.bulk_update_importance_table_based_on_online_bs(loss_map)

    def get_init_mini_batch(self, init_id):
        # init_id from 0~self.num_iters_per_epoch
        selected_sample_ids = list(range(init_id * self.batch_size, init_id * self.batch_size + self.batch_size))

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


def bytes_to_int(bytes_array):
    result = 0
    for b in bytes_array:
        result = result * 256 + int(b)
    return result