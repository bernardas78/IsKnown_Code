from sys import platform
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input

class Glb:
    #images_folder = '/home/bernardas/IsKnown_Images' if platform=='linux' else 'C:/IsKnown_Images_IsVisible'

    if platform=='linux':
        images_folder = '/home/bernardas/IsKnown_Images'
        results_folder = '/home/bernardas/IsKnown_Results'
        tensorboard_logs_folder = '/home/bernardas/IsKnown_TBLogs'
        cache_folder = '/home/bernardas/IsKnown_Cache'
        batch_size=256
    else:
        images_folder = 'C:/IsKnown_Images_IsVisible'
        results_folder = 'C:/IsKnown_Results'
        tensorboard_logs_folder = 'C:/IsKnown_TBLogs'
        cache_folder = 'C:/IsKnown_Cache'
        batch_size = 1024

class Glb_Iterators:
    def get_iterator (data_folder, div255_resnet, batch_size=32, target_size=256):
        dataGen = ImageDataGenerator(
            rescale= None if div255_resnet!="div255" else 1./255,
            preprocessing_function= None if div255_resnet!="resnet" else resnet_preprocess_input)

        real_target_size = target_size if div255_resnet!="resnet" else 224

        data_iterator = dataGen.flow_from_directory(
            directory=data_folder,
            target_size=(real_target_size, real_target_size),
            batch_size=batch_size,
            shuffle=True,
            class_mode='categorical')

        return data_iterator