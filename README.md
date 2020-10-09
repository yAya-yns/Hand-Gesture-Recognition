Gesture detection - We implement a model to distinguish between a large class of hand gestures, using . This model will consist of a keypoint detector to generate a small array of keypoints, followed by another model to distinguish the gestures based on the keypoints. The model will be trained using a public hand gesture dataset, such as sdhttps://www.kaggle.com/gti-upm/leapgestrecog?
 Examples:
	Fully connected model
	GNN
	VAE
Possible Extended Goals:
	Train the model end-to-end, letting the model itself pick the keypoints. 
	Using a siamese network and VAE, allow for few-shot learning of custom gestures
	Speed up inference to run real-time (>5FPS)
