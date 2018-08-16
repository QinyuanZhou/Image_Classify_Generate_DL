import tensorflow as tf

class Conv_AE(object):
    def __init__(self, input_shape=(None, 96, 96, 3), learning_rate=0.001):
        self.img_shape = input_shape
        self.img_width = input_shape[1]
        self.img_height = input_shape[2]
        self.img_depth = input_shape[3]
        self.X = tf.placeholder(dtype=tf.float32, shape=input_shape)
        self.rand_code = tf.placeholder(dtype=tf.float32, shape=(None, 6, 6, 16))

        self.learning_rate = learning_rate

        self.c_stride = [1, 1, 1, 1]
        self.p_size = [1, 2, 2, 1]
        self.p_stride = [1, 2, 2, 1]

    def encoder(self):
        with tf.name_scope('Encoder_Layer'):
            with tf.name_scope('Conv_Layer1'):
                conv1 = tf.nn.conv2d(self.X, filter=tf.Variable(
                    tf.truncated_normal([3, 3, self.img_depth, 64], stddev=0.1, dtype=tf.float32), name='weight1'),
                                     strides=self.c_stride, padding='SAME') # n x 96 x 96 x 63
                relu1 = tf.nn.relu(conv1 + tf.Variable(tf.constant(0.1, shape=[64]), name='bias1'))
                pool1 = tf.nn.max_pool(relu1, ksize=self.p_size, strides=self.p_stride, padding='SAME') # n x 48 x 48 x 64

            with tf.name_scope('Conv_Layer2'):
                conv2 = tf.nn.conv2d(pool1, filter=tf.Variable(
                    tf.truncated_normal([3, 3, 64, 32], stddev=0.1, dtype=tf.float32), name='weight2'),
                                     strides=self.c_stride, padding='SAME') # n x 48 x 48 x 32
                relu2 = tf.nn.relu(conv2 + tf.Variable(tf.constant(0.1, shape=[32]), name='bias2'))
                pool2 = tf.nn.max_pool(relu2, ksize=self.p_size, strides=self.p_stride, padding='SAME') # n x 24 x 24 x 32

            with tf.name_scope('Conv_Layer3'):
                conv3 = tf.nn.conv2d(pool2, filter=tf.Variable(
                    tf.truncated_normal([3, 3, 32, 16], stddev=0.1, dtype=tf.float32), name='weight3'),
                                     strides=self.c_stride, padding='SAME') # n x 24 x 24 x 16
                relu3 = tf.nn.relu(conv3 + tf.Variable(tf.constant(0.1, shape=[16]), name='bias3'))
                self.econ = tf.nn.max_pool(relu3, ksize=self.p_size, strides=self.p_stride, padding='SAME') # n x 12 x 12 x 16

                p_mean, p_var = tf.nn.moments(self.econ, axes=[0, 1, 2], name='E_moments1')
                scale = tf.Variable(tf.ones([1]))
                shift = tf.Variable(tf.zeros([1]))
                epsilon = 0.000001
                self.econ = tf.nn.batch_normalization(self.econ, p_mean, p_var, scale, shift, epsilon, name='E_batch_normalization')

    def middle_code(self): # n x 12 x 12 x 16
        with tf.name_scope('Middle_Code'):
            with tf.name_scope('M_Layer1'):
                m_conv1 = tf.nn.conv2d(self.econ, filter=tf.Variable(
                    tf.truncated_normal([3, 3, 16, 16], stddev=0.1, dtype=tf.float32), name='M_weight1'),
                                     strides=self.c_stride, padding='SAME') # n x 12 x 12 x 16
                m_relu1 = tf.nn.relu(m_conv1 + tf.Variable(tf.constant(0.1, shape=[16]), name='M_bias1'))
                self.m_code = tf.nn.max_pool(m_relu1, ksize=self.p_size, strides=self.p_stride, padding='SAME') # n x 6 x 6 x 16

                m_mean, m_var = tf.nn.moments(self.m_code, axes=[0, 1, 2], name='M_moments')
                scale = tf.Variable(tf.ones([1]))
                shift = tf.Variable(tf.zeros([1]))
                epsilon = 0.000001
                self.m_code = tf.nn.batch_normalization(self.m_code, m_mean, m_var, scale, shift, epsilon, name='M_batch_normalization')

    def decoder(self, rand_code): # n x 6 x 6 x 16
        with tf.name_scope('Decoder_Layer'):
            with tf.name_scope('Decoder_Layer1'):
                d_conv1 = tf.image.resize_nearest_neighbor(rand_code, (12, 12)) # n x 12 x 12 x 16
                d_conv1 = tf.nn.conv2d(d_conv1, filter=tf.Variable(
                    tf.truncated_normal([3, 3, 16, 16], stddev=0.1, dtype=tf.float32), name='D_weight1'),
                                     strides=self.c_stride, padding='SAME') # n x 12 x 12 x 16
                d_conv1 = tf.nn.relu(d_conv1 + tf.Variable(tf.constant(0.1, shape=[16]), name='D_bias1'))

            with tf.name_scope('Decoder_Layer2'):
                d_conv2 = tf.image.resize_nearest_neighbor(d_conv1, (24, 24)) # n x 24 x 24 x 16
                d_conv2 = tf.nn.conv2d(d_conv2, filter=tf.Variable(
                    tf.truncated_normal([3, 3, 16, 32], stddev=0.1, dtype=tf.float32), name='D_weight2'),
                                     strides=self.c_stride, padding='SAME') # n x 24 x 24 x 32
                d_conv2 = tf.nn.relu(d_conv2 + tf.Variable(tf.constant(0.1, shape=[32]), name='D_bias2'))

            with tf.name_scope('Decoder_Layer3'):
                d_conv3 = tf.image.resize_nearest_neighbor(d_conv2, (48, 48)) # n x 48 x 48 x 32
                d_conv3 = tf.nn.conv2d(d_conv3, filter=tf.Variable(
                    tf.truncated_normal([3, 3, 32, 64], stddev=0.1, dtype=tf.float32), name='D_weight3'),
                                     strides=self.c_stride, padding='SAME') # n x 48 x 48 x 64
                d_conv3 = tf.nn.relu(d_conv3 + tf.Variable(tf.constant(0.1, shape=[64]), name='D_bias3'))

            with tf.name_scope('Decoder_Layer4'):
                d_conv4 = tf.image.resize_nearest_neighbor(d_conv3, (96, 96)) # n x 96 x 96 x 32
                recon = tf.nn.conv2d(d_conv4, filter=tf.Variable(tf.truncated_normal([3, 3, 64, 3], stddev=0.1,
                                dtype=tf.float32), name='D_weight4'), strides=self.c_stride, padding='SAME') # n x 96 x 96 x 3
        return recon

    def train_net(self, sess, X):
        feed_dict = {self.X: X}
        self.encoder()
        self.middle_code()
        recon_image = self.decoder(self.m_code)
        self.loss = tf.reduce_mean(tf.square(self.X - recon_image))
        train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        model_saver = tf.train.Saver()
        init_var = tf.global_variables_initializer()
        sess.run(init_var)
        r, l, t = sess.run([recon_image, self.loss, train], feed_dict=feed_dict)

        return r, l, t, model_saver

    def random_generate_image(self, sess, rand_code):
        feed_dict = {self.rand_code: rand_code}
        generate_image = self.decoder(rand_code)
        
        return sess.run(generate_image, feed_dict=feed_dict)

