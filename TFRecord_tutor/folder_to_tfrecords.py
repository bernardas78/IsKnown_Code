import tensorflow as tf
from Globals.globalvars import Glb_Iterators, Glb
import os
from PIL import Image
import numpy as np
import base64
import time

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytesS_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _floatS_feature(values):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

hier_lvl=0
set_name="Test"
batch_size=32
div255_resnet = "div255"
img_filepath = os.path.join( Glb.images_folder, "Bal_v14", "Ind-{}".format(hier_lvl), set_name)
data_iterator = Glb_Iterators.get_iterator(img_filepath, div255_resnet=div255_resnet, batch_size=batch_size)

# all file names to list
allfiles_path = []
for barcode_path in os.listdir(img_filepath):
    allfiles_path += [ os.path.join(img_filepath,barcode_path,filepath) for filepath in os.listdir( os.path.join (img_filepath,barcode_path )) ]

now= time.time()
for i,(X,y) in enumerate(data_iterator):
    #print ("batch {}/{}".format(i,len(data_iterator)))
    #if i+1>=0: #len(data_iterator):
    if i + 1 >= len(data_iterator):
        break
print ("Time elapsed: {}sec".format(time.time()-now))
#Time elapsed: 23.132225275039673sec

feature = {
      'img': _floatS_feature(X[0].flatten()),
      'height': _int64_feature(X[0].shape[0]),
      'width': _int64_feature(X[0].shape[1]),
      'lbl': _int64_feature(np.argmax(y[0]))
  }
example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
example_serialized = example_proto.SerializeToString()

def generator():
    for filename in allfiles_path:
        with open(filename, 'rb') as f:
            # The file content is a jpeg encoded bytes object
            in_jpg_encoding = f.read()
            image_raw = base64.b64encode(in_jpg_encoding)

        #img = Image.open(filename)
        #image_raw = img.tostring()
            feature = {
                  'img_raw': _bytes_feature(image_raw),
                  'filename': _int64_feature(11)
              }
            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
            example_serialized = example_proto.SerializeToString()
            yield example_serialized

#serialized_features_dataset = features_dataset.map(tf_serialize_example)
#def generator_iterator():
#    for i, (X, y) in enumerate(data_iterator):
#        print("batch {}/{}".format(i, len(data_iterator)))
#        height,width = X.shape[1:3]
#        for sample_id in range(X.shape[0]):
#            feature = {
#                  'img': _floatS_feature(X[sample_id].flatten()),
#                  'height': _int64_feature(height),
#                  'width': _int64_feature(width),
#                  'lbl': _int64_feature(np.argmax(y[sample_id]))
#              }
#            example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
#            example_serialized = example_proto.SerializeToString()
#            yield example_serialized
#        if i + 1 >= len(data_iterator):
#            break

serialized_features_dataset = tf.data.Dataset.from_generator( generator, output_types=tf.string, output_shapes=())

filename = os.path.join(Glb.results_folder, 'file.tfrecords')
writer = tf.data.experimental.TFRecordWriter(filename)
if False:
    writer.write(serialized_features_dataset)

# Retrieval
now= time.time()
i=0
raw_dataset = tf.data.TFRecordDataset([filename]).batch(32)
for raw_record in raw_dataset:
    #print ("batch {}".format(i))
    i+=1
    #print(repr(raw_record))
    #example_proto = tf.train.Example.FromString(raw_record)
print ("Time elapsed: {}sec".format(time.time()-now))
#Time elapsed: 5.329460859298706sec