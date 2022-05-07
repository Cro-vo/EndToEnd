# Convert image to pure black and white
# Show usages:
# python tobw.py -h
# python tobw.py -i rain.svg -o rain_bw.jpg  -s 12x12 -t 200
from PIL import Image, UnidentifiedImageError
import argparse
import numpy as np
from cairosvg import svg2png
import os
from random import random

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input-image", help="Input image", type=str, required=True)
ap.add_argument("-o", "--output-image", help="output image", type=str, required=True)
ap.add_argument("-t", "--threshold",
                help="(Optional) Threshold value, a number between 0 and 255", type=int, default=128)
ap.add_argument("-r", "--invert-bw",
                help="(Optional) Invert black and white", action='store_true')
ap.add_argument("-s", "--size", help="(Optional) Resize image to the specified size, eg., 16x16", type=str)

args = vars(ap.parse_args())

invert_bw = args['invert_bw']
threshold = args['threshold']
if threshold > 255 or threshold < 0:
    raise Exception('Invalid threshold value!')
img_in = args['input_image']
img_out = args['output_image']

if args['size']:
    size = [int(n) for n in args['size'].lower().split('x')]
else:
    size = None

high = 255
low = 0
if invert_bw:
    high = 0
    low = 255

def replace_ext(filename, new_ext):
    ext = filename.split('.')[-1] if '.' in filename else ''
    return filename[:len(filename)-len(ext)] + str(new_ext)


def remove_transparency(im, bg_colour=(255, 255, 255)):
    # https://stackoverflow.com/questions/35859140/remove-transparency-alpha-from-any-image-using-pil/35859141
    # Only process if image has transparency (http://stackoverflow.com/a/1963146)
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
        alpha = im.convert('RGBA').getchannel('A')
        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg
    else:
        return im


# color image

try:
    col = Image.open(img_in)
except UnidentifiedImageError:
    if (img_in.lower().endswith('.svg')):
        tmp = replace_ext(img_in, '{}.png'.format(random()))
        svg2png(url=img_in, write_to=tmp)
        col = Image.open(tmp)
    else:
        raise Exception('unknown image type')

if size:
    col = col.resize(size)
col = remove_transparency(col)
gray = col.convert('L')
bw = gray.point(lambda x: low if x < threshold else high, '1')
bw.save(img_out)

for u8 in np.uint8(bw):
    print(''.join(str(c) for c in u8))

print()
print('Result as XBM image:')
if (img_out.lower().endswith('.xbm')):
    print(open(img_out).read())
else:
    xbm_out = replace_ext(img_in, '{}.xbm'.format(random()))
    bw.save(xbm_out)
    print(open(xbm_out).read())
    os.remove(xbm_out)

if tmp is not None:
    os.remove(tmp)