import glob
from PIL import Image


i = 0
for filename in glob.glob('full_set/*.jpg'):
    original_image = Image.open(filename)
    cropped_image = original_image.crop((850, 475, 1625, 1275))
    cropped_image.save('cropped_full_set/' + str(i) + '.jpg')
    i += 1

    print(i)

    # 850 375 1625 1475
