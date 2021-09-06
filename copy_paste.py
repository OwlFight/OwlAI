from PIL import Image
from glob import glob
from tqdm import tqdm

choosed_folder = glob("train_image/*")
print(choosed_folder[0])

original_folder = glob("original_images/*")
print(original_folder[0])

resized_folder = 'resized_images'

image_number = []
image_directory = []

for image_name in choosed_folder:
	image_number.append(image_name.split("\\")[1])

for image_name in original_folder:
	if image_name.split("\\")[1] == image_number[0]:
		image_directory.append(image_name)
		

print(image_number)