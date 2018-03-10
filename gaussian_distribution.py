import tensorflow as tf
import numpy as np

raw_data = np.random.normal(100,50,10000)

alpha = tf.constant(0.05)
current_value = tf.placeholder(tf.float32)
prev_avg = tf.Variable(0.0)
update_avg = alpha * current_value + (1 - alpha) * prev_avg

avg_hist = tf.summary.scalar("running_average", update_avg)
value_hist = tf.summary.scalar("incomming_values", current_value)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs")
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for i in range(len(raw_data)):
		summary_str, current_avg = sess.run([merged, update_avg], feed_dict={current_value: raw_data[i]})
		sess.run(tf.assign(prev_avg, current_avg))
		print(raw_data[i], current_avg)
		writer.add_summary(summary_str, i)
