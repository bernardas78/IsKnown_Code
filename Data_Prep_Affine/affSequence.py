
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2 as cv
import numpy as np

class affSequence (ImageDataGenerator):


    def __init__(self, variation=20, **kwargs):
        super().__init__(**kwargs)
        self.variation = variation

    def apply_transform(self, x, transform_parameters):
        #print ("Inside transform. x.shape: {}".format(x.shape))

        fun_rand_pt = lambda : np.random.randint(low=-self.variation, high=self.variation+1)

        src_pts = np.array([
            [0          + fun_rand_pt(), 0          + fun_rand_pt()],
            [x.shape[0] + fun_rand_pt(), 0          + fun_rand_pt()],
            [0          + fun_rand_pt(), x.shape[1] + fun_rand_pt()],
            [x.shape[0] + fun_rand_pt(), x.shape[1] + fun_rand_pt() ]]).astype(np.float32)

        dst_pts = np.array([
            [0          + fun_rand_pt(), 0          + fun_rand_pt()],
            [x.shape[0] + fun_rand_pt(), 0          + fun_rand_pt()],
            [0          + fun_rand_pt(), x.shape[1] + fun_rand_pt()],
            [x.shape[0] + fun_rand_pt(), x.shape[1] + fun_rand_pt() ]]).astype(np.float32)

        # Affine transform
        M = cv.getPerspectiveTransform(src_pts, dst_pts)
        transformed_x = cv.warpPerspective(x, M, x.shape[0:2])

        return transformed_x
