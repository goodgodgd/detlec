import tensorflow as tf

# create Tensor from List with specific type
x = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=tf.int32)

print("Type of every element:", x.dtype)
print("Number of axes (=Rank):", x.ndim)
print("Shape of tensor:", x.shape)
print("Total number of elements: ", tf.size(x))

print("To numpy array:\n", x.numpy()[0])
print("Back to Tensor:", tf.convert_to_tensor(x.numpy())[0])


def print_tensor_shape(tensor, title):
    print(f"{title} 1) Tensor.shape:", tensor.shape, type(tensor.shape))
    print(f"{title} 2) Tensor.get_shape():", tensor.get_shape())
    print(f"{title} 3) tf.shape():", tf.shape(tensor))
    h, w = tensor[0, 0, 1], tensor[0, 0, 2]
    zeros = tf.zeros((h, w))
    print(f"{title} 4) Tensor.shape:", zeros.shape)
    print(f"{title} 5) Tensor.get_shape():", zeros.get_shape())
    print(f"{title} 6) tf.shape():", tf.shape(zeros))
    return tf.shape(zeros)


@tf.function
def print_tensor_shape_graph(tensor, title):
    return print_tensor_shape(tensor, title)


# Shape: The length (number of elements) of each of the axes of a tensor
print_tensor_shape(x, "eager")
shape6 = print_tensor_shape_graph(x, "graph")
print("graph 6-1) shape:", shape6)
shape6 = print_tensor_shape_graph(x, "graph")
shape6 = print_tensor_shape_graph(x, "graph")
