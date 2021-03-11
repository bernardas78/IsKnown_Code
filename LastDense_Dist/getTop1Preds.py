# Compare Top1 prediction distributions between known and unknown classes
#   Model trained for 59 classes (SCO 22-24)

from keras.models import load_model
import pandas as pd
import cv2 as cv
import numpy as np

df_probs_filename = "top1_preds.csv"

def prepareImage (filename, target_size):
	# Load image ( BGR )
	img = cv.imread(filename)
	# convert to RGB
	img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
	# Resize to target
	img = cv.resize ( img, (target_size,target_size) )
	# Subtract global dataset average to center pixels to 0
	img = img / 255.
	return img

model_filename = "A:/RetellectModels/model_20201121_59prekes.h5"
model = load_model(model_filename)

# Load list of filenames
df_filenames = pd.read_csv("ListFiles.csv", header=None, names=["filename"])

# Empty results file
column_names = ["IsKnown", "Top1_prob"]
df_probs = pd.DataFrame (columns=column_names )
# Uncomment to overwrite
df_probs.to_csv(df_probs_filename, index=False, header=True, mode='w')

target_size = model.layers[0].input_shape[1]

# Predict top 1 and store in a file
for filename in df_filenames["filename"]:
	img = prepareImage(filename,target_size)
	X = np.stack([img])  # quick way to add a dimension
	probs = model.predict(X)
	top1_prob = np.max(probs)

	#print (filename)
	IsKnown = "Unknown" if (filename.split("\\")[-2]=="UnKnown") else "Known"
	df_probs = pd.DataFrame(
		data=[np.hstack([IsKnown, top1_prob])],
		columns=column_names)
	df_probs.to_csv(df_probs_filename, header=None, index=None, mode='a')
