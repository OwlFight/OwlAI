from PIL import Image
from glob import glob
from tqdm import tqdm

image_list = glob("train_image/*")

for image_name in tqdm(image_list):

	image = Image.open(image_name)
	image = image.convert('RGB')
	image = image.resize((100,100))
	image.save(image_name)

print("Done!")