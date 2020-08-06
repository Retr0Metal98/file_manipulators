'''
Module with functions used to manipulate image samples generated using HyperGAN
'''
import os
from PIL import Image
import matplotlib.pyplot as plt
def hypergan_sample_separator(source_img,sample_width,sample_height,dest_folder,save_mode=True): 
    # produces separate files
    # sample_width & sample_height are in pixels
    im = Image.open(source_img)
    im_width,im_height = im.size
    cols = im_width//sample_width
    rows = im_height//sample_height
    x, y = 0, 0
    i = 1
    im_list = []
    while y < im_height:
        crop_area = (x,y,x+sample_width,y+sample_height)
        sl = im.crop(crop_area)
        sl.load()
        im_list.append(sl)
        if save_mode:
            sl.save(os.path.join(dest_folder,"Image_{0}.png".format(i)))
            # print("Saved image ", i)
        x += sample_width
        if x == im_width:
            x = 0
            y += sample_height
        
        i += 1
    if not save_mode:
        return im_list

def hypergan_sample_rearranger(source_img,sample_width,sample_height,gap_width,dest_img,selection_mode=False):
    # joins the images into a single line; with `gap_width` black pixels between each
    im = Image.open(source_img)
    imgs = hypergan_sample_separator(source_img,sample_width,sample_height,"",save_mode=False)
    im_width,im_height = im.size
    rows, cols = im_height // sample_height, im_width // sample_width
    img_ids = []
    if selection_mode:
        print("Rearranging sample:", source_img)
        plt.figure(figsize=(20,20))
        for n in range(len(imgs)):
            ax = plt.subplot(rows,cols,n+1)
            plt.imshow(imgs[n])
            plt.title("Image ID "+str(n))
            plt.axis('off')
        img_ids = []
        finish_choosing = False
        print("Please enter IDs of images to include in merged copy (type 'N' to finish):")
        while not finish_choosing:
            inp = input("\n")
            if inp == 'N':
                finish_choosing = True
            try:
                img_ids.append(int(inp))
            except:
                pass
    else:
        img_ids = range(len(imgs))
    
    compiled_width, compiled_height = int(len(img_ids)*sample_width) + int((len(img_ids)-1)*gap_width), sample_height
    compiled_img = Image.new('RGB',(compiled_width, compiled_height))
    
    x,y = 0,0
    for i in img_ids:
        im = imgs[i]
        compiled_img.paste(im,(x,y))
        x += sample_width+gap_width
    
    compiled_img.save(dest_img)
