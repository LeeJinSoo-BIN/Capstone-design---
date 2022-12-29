import cv2
import scipy.ndimage.morphology as sm
import skimage.morphology as mp
from skimage.segmentation import clear_border
# #img_name = '000010_0.png'
# img_name = '004325_1.jpg'

# # load_path = './data/test/image/'+img_name
# # save_path = './data/test/image-mask/'+img_name[:-3]+'png'
# load_path = './data/test/cloth/'+img_name
# load_path2 = './data_ori/test/cloth/'+img_name
# save_path = './data/test/cloth-mask/'+img_name[:-3]+'png'
# img = cv2.imread(load_path, 0)
# th, img1 = cv2.threshold(img, 194, 255, cv2.THRESH_BINARY_INV)
# img2 = sm.binary_fill_holes(img1)
# img3 = (img2*255).astype('uint8')


#cv2.imwrite(save_path, img3)



def masking_img(img) :
    th, img1 = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY_INV)
    img2 = sm.binary_fill_holes(img1)
    img3 = (img2*255).astype('uint8')

    return img3