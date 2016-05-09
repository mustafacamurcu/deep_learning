NUM_IMAGES = 100000
from PIL import Image
from PIL import ImageDraw
from random import randrange, random

def createImage(name, color1, color2):
    img = Image.new('RGB',(227,227), color1)
    img.load()
    draw = ImageDraw.Draw(img)
    rx = randrange(60) + 20
    ry = rx * (.8 + randrange(40) / 100.)
    x = (227 - 2 * rx) * random() + rx
    y = (227 - 2 * ry) * random() + ry
    xmin = max(0, int(x - rx))
    xmax = max(0, int(x + rx))
    ymin = min(226, int(y - ry))
    ymax = min(226, int(y + ry))
    draw.ellipse((xmin, ymin, xmax, ymax), color2)
    img.save("/afs/csail.mit.edu/u/k/kocabey/Desktop/CircleData/Images/" + name + ".png", "PNG")
    f = open("/afs/csail.mit.edu/u/k/kocabey/Desktop/CircleData/Annotations/" + name + ".txt", "w")
    f.write(str(xmin) + "\n" + str(ymin) + "\n" + str(xmax) + "\n" + str(ymax))
    f.close()

for i in range(NUM_IMAGES):
    if i % 10000 == 0:
        print i	
    createImage("image_" + str(i), randrange(0xffffff), randrange(0xffffff))
