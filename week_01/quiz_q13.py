from PIL import Image, ImageShow, ImageFilter
import math

im = Image.open("bossier.jpg")

print im.format, im.size, im.mode


# im.show()

# d = 128
# out = im.point(lambda i: d * math.floor(i/d))
# out.show()

# out2 = im.filter(ImageFilter.MedianFilter(3))
# out2.show()

scale = 5
out3 = im.resize((int(im.width/scale), int(im.height/scale)))
out3.show()
