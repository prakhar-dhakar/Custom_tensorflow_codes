import matplotlib.image as mpimg
from keras.utils import to_categorical
 
def generator(samples, directory, batch_size=32,target_size = (224,224)):

    def resize_img_keeping_aspect(image,target_size):
        # print(image.shape)
        x,y = target_size
        img = tf.image.resize_with_pad(image,
                                        x,
                                        y,
                                        method=tf.image.ResizeMethod.BILINEAR
                                        )
        return img

    def preprocess(image):
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.2, 0.5)
        image = tf.image.random_flip_left_right(image)
        return image/255.0


    """
    Yields the next training batch.
    Suppose `samples` is an array [[image1_filename,label1], [image2_filename,label2],...].
    """
    num_samples = len(samples)
    input_x, input_y = target_size
    num_steps = num_samples/batch_size
    num_labels = samples[0][1].shape[0]

    while True: # Loop forever so the generator never terminates
        shuffle(samples)
 
        # Get index to start each batch: [0, batch_size, 2*batch_size, ..., max multiple of batch_size &lt;= num_samples]
        for offset in range(0, num_samples, batch_size):
            # Get the samples you'll use in this batch
            batch_samples = samples[offset:offset+batch_size]
            # Initialise X_train and y_train arrays for this batch
            this_batch_size = len(batch_samples)
            X_train = np.zeros(shape = (this_batch_size,input_x,input_y,3))
            y_train = np.zeros(shape = (this_batch_size,num_labels))

            count = 0
 
            # For each example
            for batch_sample in batch_samples:
                # Load image (X)
                filename =  directory +batch_sample[0]
                image = mpimg.imread(filename)
                image = resize_img_keeping_aspect(image, target_size)
                # Read label (y)
                y = batch_sample[1]
                # Add example to arrays
                X_train[count, :] = image
                y_train[count, :] = y
                count+=1
 
            # Make sure they're numpy arrays (as opposed to lists)
            # Preprocessing the images as a batch using tf.image
            X_train = preprocess(X_train)
 
            # The generator-y part: yield the next training batch            
            yield X_train.numpy(), y_train
 
