import tensorflow as tf

n_classes = 2
img_dim = 48
n_channels = 3
adience_img_feature_width = 64
img_feature_width = 0
num_of_units = 5
n_inputs = 20  # word vector dimension
n_steps = 80  # number of words fed to each RNN. We will feed the profile and tweet together
text_feature_length = n_steps * n_inputs
profile_color_feature_length = 0
#feature_width = img_feature_width + text_feature_length + profile_color_feature_length
feature_width = text_feature_length
adience_keep_prob = tf.placeholder(tf.float32)


best_accuracy_so_far = 0.0

x = tf.placeholder(tf.float32, shape=[None, img_dim, img_dim, n_channels])
y_ = tf.placeholder(tf.int32, shape=[None, n_classes])


rnn_X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
rnn_other_features = tf.placeholder(tf.float32, shape=[None,0])
n_neurons = 300
learning_rate = 0.01
