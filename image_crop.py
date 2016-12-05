import glob
from PIL import Image

def splitFilename(filename):
    res = filename.split("/")
    return res[1]

useFullSet = False
path  = ""
cropped_path = ""

if (useFullSet):
    path = "full_set"
    cropped_path = "cropped_full_set"
else:
    path = "me"
    cropped_path = "cropped_me"


i = 0
for filename in glob.glob(path + '/*.jpg'):
    name = splitFilename(filename)
    original_image = Image.open(filename)
    width, height = original_image.size
    left, top, right, bottom = int(width*0.345), int(height*0.276), int(width*0.665), int(height*0.742)
    cropped_image = original_image.crop((left, top, right, bottom))
    cropped_image.save(cropped_path + '/' + name)
    i += 1

    print(i)

    # 850 375 1625 1475
