import tensorflow as tf


class Model:
    def __init__(self, x, label, position_scores, position_labels):
        #self.batch_size = batch_size
        self.prediction  = self.block_net(x, False)
        # print(self.prediction.shape)
        # print(label.shape)
        self.loss, self.loss_pos, self.loss_wgh = self.losses(position_labels, position_scores, self.prediction, label)
    image_size =  16#self.frame_hr, self.frame_sr,, self.g_frame_loss, self.d_frame_loss
    
    def block_net(self, x, reuse):#3854.68kb
        with tf.variable_scope('block_net', reuse=reuse):
            dense1 = tf.layers.dense(inputs=x, 
                      units=256,#256
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.nn.l2_loss)#initializers.random_normal,tf.truncated_normal_initializer(stddev=0.01)
            dense2= tf.layers.dense(inputs=dense1, 
                      units=256,#256
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.nn.l2_loss)
            dense3= tf.layers.dense(inputs=dense2,
                      units=64,#64
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.nn.l2_loss)
            #dense4= tf.layers.dense(inputs=dense3,
                       #units=128,
                       #activation=tf.nn.relu,
                       #kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                       #kernel_regularizer=tf.nn.l2_loss)
            dense4= tf.layers.dense(inputs=dense3,
                        units=1, 
                        activation=tf.nn.sigmoid,#tf.nn.relu,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.nn.l2_loss)
            
        self.block1_net_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope='block_net')
        return dense4

    def losses(self, position_labels, position_scores, prediction, label):
        
        def inference_content_loss_sr(frame_hr, frame_sr):
            content_base_loss = tf.reduce_mean(tf.sqrt((frame_hr - frame_sr) ** 2+(1e-4)**2))
            return tf.reduce_mean(content_base_loss)
            
        def inference_content_loss_sr1(frame_hr, frame_sr):
            content_base_loss = tf.reduce_mean(tf.nn.l2_loss(frame_hr - frame_sr))
            return tf.reduce_mean(content_base_loss)

        def inference_content_loss_sr2(tf_train_labels,logits):
            #content_base_loss = tf.nn.softmax_cross_entropy_with_logits(logits, labels, dim=-1, name=None)
            content_base_loss = tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits)
            return tf.reduce_mean(content_base_loss)

        content_sr_loss_p = inference_content_loss_sr(position_labels, position_scores)
        content_sr_loss_w = inference_content_loss_sr(label, prediction)
        loss = 0.1*content_sr_loss_p + content_sr_loss_w

        return loss, content_sr_loss_p, content_sr_loss_w