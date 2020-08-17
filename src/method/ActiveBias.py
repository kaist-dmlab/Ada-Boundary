import os, sys, time
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from reader import batch_patcher as patcher
from reader import active_bias_sampler
from network.WideResNet.WideResNet_Weighted_Loss import *
from network.DenseNet.DenseNet_Weighted_Loss import *

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf.logging.set_verbosity(tf.logging.ERROR)

def ActiveBias(gpu_id, input_reader, model_type, training_epochs, batch_size, lr_boundaries, lr_values, optimizer_type, warm_up_period, smoothness=0.02, pretrain=0, log_dir="log"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    text_log = []
    text_log.append("epoch, time(s), learning rate, minibatch loss, minibatch error, test loss, test error")

    num_train_images = input_reader.num_train_images
    num_val_images = input_reader.num_val_images
    num_label = input_reader.num_classes
    image_shape = [input_reader.width, input_reader.height, input_reader.depth]

    train_batch_patcher = patcher.BatchPatcher(num_train_images, batch_size, num_label)
    validation_batch_patcher = patcher.BatchPatcher(num_val_images, batch_size, num_label)
    sampler = active_bias_sampler.Sampler(num_train_images, num_label, smoothness)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.visible_device_list = str(gpu_id)
    config.gpu_options.allow_growth = True
    graph = tf.Graph()

    with graph.as_default():
        with tf.device('/gpu:' + str(gpu_id)):
            with tf.Session(config=config) as sess:

                # Input Graph Generation #############################################################################
                t_ids, t_images, t_labels = input_reader.data_read(batch_size, train=True)
                v_ids, v_images, v_labels = input_reader.data_read(batch_size, train=False)

                # Model Graph Construction ###########################################################################
                if model_type == "DenseNet-25-12":
                    model = DenseNet(25, 12, image_shape, num_label, batch_size, batch_size)
                elif model_type == "WideResNet16-8":
                    model = WideResNet(16, 8, image_shape, num_label, batch_size, batch_size)

                train_loss_op, train_accuracy_op, train_op, train_xentropy_op, train_prob_op, train_distance_op = model.build_train_op(lr_boundaries, lr_values, optimizer_type)
                test_loss_op, test_accuracy_op, _ = model.build_test_op()

                # Data load in memeory ###############################################################################
                print("start to load data set.")
                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)
                train_batch_patcher.bulk_load_in_memory(sess, t_ids, t_images, t_labels)
                validation_batch_patcher.bulk_load_in_memory(sess, v_ids, v_images, v_labels)

                start_time = time.time()
                # Model Initialization ###########################################################################
                # init params: we share the initial epochs. See paper.
                if pretrain != 0:
                    start_time = time.time()
                    saver = tf.train.Saver()
                    file_dir = "init_weight/" + input_reader.dataset_name + "/" + model_type + "_" + optimizer_type + "_lr=" + str(lr_values[0]) + "_e=" + str(pretrain) + "/"
                    minus_start_time = 0
                    with open(file_dir + "text_log.csv") as f:
                        for line in f:
                            print(line, end="")
                            text_log.append(line)
                            minus_start_time = line.split(",")[1]
                    start_time = start_time - float(minus_start_time)
                    saver.restore(sess, file_dir + "model.ckpt")
                    print("shared weight is successfully loaded")
                else:
                    sess.run(tf.global_variables_initializer())

                # Traing Process #####################################################################################
                for epoch in range(pretrain, training_epochs):

                    # (1) Mini-batch loss and error along with netowrk updates
                    avg_mini_loss = 0.0
                    avg_mini_acc = 0.0
                    for i in range(train_batch_patcher.num_iters_per_epoch):
                        # if is_warm_up = True, then select next batch samples uniformly at random
                        ids, images, labels = train_batch_patcher.get_next_mini_batch(num_of_sample=batch_size)
                        if epoch < warm_up_period:
                            weights = sampler.compute_sample_weights(ids, uniform=True)
                        else:
                            weights = sampler.compute_sample_weights(ids)
                        mini_loss, mini_acc, _, softmax_matrix, distance = sess.run([train_loss_op, train_accuracy_op, train_op, train_prob_op, train_distance_op], feed_dict={model.train_image_placeholder: images, model.train_label_placeholder: labels, model.train_weight_placeholder: weights})
                        sampler.async_update_probability_matrix(ids, labels, softmax_matrix)
                        avg_mini_loss += mini_loss
                        avg_mini_acc += mini_acc
                    avg_mini_loss /= train_batch_patcher.num_iters_per_epoch
                    avg_mini_acc /= train_batch_patcher.num_iters_per_epoch

                    # (2) Compute training loss and error
                    avg_train_loss = 0.0
                    avg_train_acc = 0.0
                    for i in range(train_batch_patcher.num_iters_per_epoch):
                        ids, images, labels = train_batch_patcher.get_init_mini_batch(i)
                        train_loss, train_acc = sess.run([test_loss_op, test_accuracy_op], feed_dict={model.test_image_placeholder: images, model.test_label_placeholder: labels})
                        avg_train_loss += train_loss
                        avg_train_acc += train_acc
                    avg_train_loss /= train_batch_patcher.num_iters_per_epoch
                    avg_train_acc /= train_batch_patcher.num_iters_per_epoch

                    # (3) Validation (or test) loss and error
                    avg_val_loss = 0.0
                    avg_val_acc = 0.0
                    for i in range(validation_batch_patcher.num_iters_per_epoch):
                        ids, images, labels = validation_batch_patcher.get_init_mini_batch(i)
                        val_loss, val_acc = sess.run([test_loss_op, test_accuracy_op], feed_dict={model.test_image_placeholder: images, model.test_label_placeholder: labels})
                        avg_val_loss += val_loss
                        avg_val_acc += val_acc
                    avg_val_loss /= validation_batch_patcher.num_iters_per_epoch
                    avg_val_acc /= validation_batch_patcher.num_iters_per_epoch

                    # Log Writing ####################################################################################
                    cur_lr = sess.run(model.learning_rate)
                    print((epoch + 1), ", ", int(time.time() - start_time) ,", ", cur_lr, ", ", avg_mini_loss, ", ", (1.0-avg_mini_acc), ", ", avg_train_loss, ", ", (1.0-avg_train_acc), ", ", avg_val_loss, ", ", (1.0-avg_val_acc))
                    text_log.append(str(epoch + 1) + ", " + str(int(time.time() - start_time)) + ", " + str(cur_lr) + ", " + str(avg_mini_loss) + ", " + str(1.0-avg_mini_acc) + ", " + str(avg_train_loss) + ", " + str(1.0-avg_train_acc) + ", " + str(avg_val_loss) + ", " + str(1.0-avg_val_acc))

                coord.request_stop()
                coord.join(threads)
                sess.close()

        # Log Flushing
        f = open(log_dir + "/text_log.csv", "w")
        for text in text_log:
            f.write(text + "\n")
        f.close()

