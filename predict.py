import tensorflow as tf
import numpy as np
import dataset
import csv

validation_size = 0.2
image_size = 180
num_channels = 3

test_path = '/media/youssef/682CBC432CBC0DD6/Machine_Learning/Kaggle/train_example.bson'
# test_path = '/media/youssef/682CBC432CBC0DD6/Machine_Learning/Kaggle/test.bson'


x_batch, img_names, cls, images_num = dataset.get_x_batch(test_path)

## Let us restore the saved model
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('output/kaggle-model.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((images_num, 36))

### Creating the feed_dict that is required to be fed to calculate y_pred
feed_dict_testing = {x: x_batch, y_true: y_test_images}
result = sess.run(y_pred, feed_dict=feed_dict_testing)
# result is of this format [probabiliy_of_rose probability_of_sunflower]
print(len(result))
with open('submission.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    spamwriter.writerow(['_id', 'category_id'])
    for i in range(0, len(result)):
        spamwriter.writerow([img_names[i], cls[np.argmax(result[i])]])
        print("image name=", img_names[i], " result[0]=", np.argmax(result[i]), " label=", cls[np.argmax(result[i])])

print("done")
