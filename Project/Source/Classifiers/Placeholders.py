import tensorflow as tf

n_classes = 2
img_dim = 48
n_channels = 3
img_feature_width = 64
num_of_units = 5
n_inputs = 300  # word vector dimension
n_steps = 80  # number of words fed to each RNN. We will feed the profile and tweet together
word_vec_length = 600
text_feature_length = n_steps * n_inputs
profile_color_feature_length = 6
feature_width = img_feature_width + text_feature_length + profile_color_feature_length
adience_keep_prob = tf.placeholder(tf.float32)


best_accuracy_so_far = 0.0

x = tf.placeholder(tf.float32, shape=[None, img_dim, img_dim, n_channels])
y_ = tf.placeholder(tf.int32, shape=[None, n_classes])


rnn_X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
rnn_other_features = tf.placeholder(tf.float32, shape=[None, img_feature_width + profile_color_feature_length])
n_neurons = 2
learning_rate = 0.01
