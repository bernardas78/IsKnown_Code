from sys import platform

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
        results_folder = 'A:/IsKnown_Results'
        tensorboard_logs_folder = 'C:/IsKnown_TBLogs'
        cache_folder = 'C:/IsKnown_Cache'
        batch_size = 1024
