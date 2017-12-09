import tensorflow as tf

n_classes = 2
img_dim = 48
n_channels = 3
adience_img_feature_width = 64
img_feature_width = 64

n_inputs = 20  # word vector dimension
n_steps = 80  # number of words fed to each RNN. We will feed the profile and tweet together

text_feature_length = n_steps * n_inputs
profile_color_feature_length = 0
feature_width = img_feature_width + text_feature_length + profile_color_feature_length
other_features_width = img_feature_width  + profile_color_feature_length
ensemble_feature_length = 4




best_accuracy_so_far = 0.0

x = tf.placeholder(tf.float32, shape=[None, ensemble_feature_length])
y_ = tf.placeholder(tf.int32, shape=[None, n_classes])
num_of_units = 5

rnn_X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
rnn_other_features = tf.placeholder(tf.float32, shape=[None,other_features_width])
n_neurons = 300
learning_rate = 0.01
