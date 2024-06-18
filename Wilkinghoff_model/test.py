import tensorflow as tf

# 定义两个向量
vector1 = tf.constant([2, 3, 4, 5])
vector2 = tf.constant([8, 9])

# 在-1维度上堆叠
stacked_tensors = tf.stack([vector1, vector2], axis=-1)

# 打印结果
print(stacked_tensors)