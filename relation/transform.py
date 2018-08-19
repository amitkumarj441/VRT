import random, math
from PIL import Image
from pathlib import Path
from matplotlib import patches, patheffects,pyplot as plt

def center_crop(img, boxes, size):
    
    w, h = img.size
    ow, oh = size
    i = int(round(h-oh)/2)
    j = int(round(w-ow)/2)
    img = img.crop((j, i, j+ow, i+oh))
    boxes -= np.array([j,i,j,i])
    boxes[boxes<0.0]=0.0
    boxes[boxes>1.0]=1.0
#     boxes[:,0::2].clamp_(min=0, max=ow-1)
#     boxes[:,1::2].clamp_(min=0, max=oh-1)
    return np.array(img)/255.0, boxes

def resize(img, boxes, size=[416,416], max_size=1000):
#     w,h = img.size
#     if isinstance(size, int):
#         size_min = min(w,h)
#         size_max = max(w,h)
#         sw = sh = float(size)/ size_min
#         if sw*size_max * max_size:
#             sw = sh = float(max_size) / size_max
#         ow = int(w*sw + 0.5)
#         oh = int(h*sh + 0.5)
#     else:
#         ow, oh = size
#         sw = float(ow)/w
#         sh = float(oh)/h
    resized = img.resize((size[0], size[1]))
    return np.array(resized)/255.0
#     return img.resize((ow,oh), Image.BILINEAR), boxes*torch.Tensor([sw,sh,sw,sh])

def random_noise(img, boxes):
    noise = np.random.normal(0, 0.01, [img.size(1), img.size(0), 3])
    img = np.array(img)/255.0
    img+=noise
    return img

def random_flip(img, boxes):
    '''Randomly flip the given PIL Image.
    Args:
        img: (PIL Image) image to be flipped.
        boxes: (tensor) object boxes, sized [#ojb,4].
    Returns:
        img: (PIL.Image) randomly flipped image.
        boxes: (tensor) randomly flipped boxes.
    '''
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        w = img.width
        xmin = w - boxes[:,2]
        xmax = w - boxes[:,0]
        boxes[:,0] = xmin
        boxes[:,2] = xmax
    return np.array(img), boxes

def random_crop(img, boxes):
    '''Crop the given PIL image to a random size and aspect ratio.
    A crop of random size of (0.08 to 1.0) of the original size and a random
    aspect ratio of 3/4 to 4/3 of the original aspect ratio is made.
    Args:
      img: (PIL.Image) image to be cropped.
      boxes: (tensor) object boxes, sized [#ojb,4].
    Returns:
      img: (PIL.Image) randomly cropped image.
      boxes: (tensor) randomly cropped boxes.
    '''
    success = False
    for attempt in range(10):
        area = img.size[0] * img.size[1]
        target_area = random.uniform(0.56, 1.0) * area
        aspect_ratio = random.uniform(3. / 4, 4. / 3)

        w = int(round(math.sqrt(target_area * aspect_ratio)))
        h = int(round(math.sqrt(target_area / aspect_ratio)))

        if random.random() < 0.5:
            w, h = h, w

        if w <= img.size[0] and h <= img.size[1]:
            x = random.randint(0, img.size[0] - w)
            y = random.randint(0, img.size[1] - h)
            success = True
            break

    # Fallback
    if not success:
        w = h = min(img.size[0], img.size[1])
        x = (img.size[0] - w) // 2
        y = (img.size[1] - h) // 2

    img = img.crop((x, y, x+w, y+h))
    boxes -= np.array([x,y,x,y])
    boxes[boxes<0.0]=0.0
    boxes[boxes>1.0]=1.0
#     boxes[:,0::2].clamp_(min=0, max=w-1)
#     boxes[:,1::2].clamp_(min=0, max=h-1)
    return img, boxes
