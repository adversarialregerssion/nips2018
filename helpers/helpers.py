import os
import math
import numpy as np
import datetime
import tensorflow as tf
import  datetime
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy import misc
import scipy.ndimage
import scipy.misc

"""
Norms and other mathematical expression
"""
def l2_norm(x):
    x = np.array(x).flatten()
    return np.sqrt( np.sum(np.square(x)) )

def mse(X, Y):
    X = np.array(X)
    Y = np.array(Y)
    return np.mean( (X.flatten() - Y.flatten())**2. )

def psnr(mse, pixel_dynamic_range=1.):
    """
    Peak Signal to Noise Ratio
    """
    return -10 * np.log10(mse/pixel_dynamic_range)

def get_now_date():
    """
    Get current time
    """
    d = datetime.datetime.today()
    return "%s.%s.%s-%s:%s:%s" % (d.year, d.month, d.day, d.hour, d.minute, d.second)

def fast_grad(dX, epsilon):
    """
    Fast Grient closed form solution
    """
    return -epsilon * np.sign(dX)

def get_time():
    """
    Get current time
    """
    d = datetime.datetime.today()
    return "%s.%s.%s-%s:%s:%s" % (d.year, d.month, d.day, d.hour, d.minute, d.second)

"""
Color Conversions
"""
def rgb2ycbcr(X):
    for i in range(len(X)):
        X[i,:,:,:] = np.array(Image.fromarray(np.uint8(X[i,:,:,:])).convert('YCbCr'))
    return X

def ycbcr2rgb(X):
    for i in range(len(X)):
        xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
        rgb = X[i, :, :, :].astype(np.float)
        rgb[:, :, [1, 2]] -= 128
        rgb = rgb.dot(xform.T)
        np.putmask(rgb, rgb > 255, 255)
        np.putmask(rgb, rgb < 0, 0)
        X[i, :, :, :] = np.uint8(rgb)
    return X

def rgb2y(X):
    y_image = np.array(rgb2ycbcr(np.array(X)))
    y_image = y_image[0,:,:,:1]

    print(y_image.shape)

    return y_image

def convert_image(input_image, desired_size):
    """
    Converts a grayscale image (numpy array) to maximum desired_size (integer).
    Maximum value of the input grayscale image is 1. Output padding to the square
    image provides white borders. Input image needs to only have a width and height.
    PIL is only able to manage different color channels using RGB.
    """
    im = image = Image.fromarray(np.uint8(input_image*255), 'L')
    old_size = im.size  # old_size[0] is in (width, height) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)
    new_im = Image.new("L", (desired_size, desired_size), "white")

    new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))
    # new_im.show()

    output_image = np.array(new_im)/255.
    return output_image

def merge_color_channels(params, v, _w, _fY, _fooledY):
    """
    Merges output color channels with the luminosity from the input image.
    """
    # _v = np.concatenate((_v,v[:,:,1:]),axis=2)
    _w = np.concatenate((_w,v[:,:,1:]),axis=2)

    _fY = np.reshape(_fY, [params["image_dims"][0], params["image_dims"][1], 2])
    _fY = np.concatenate((v[:,:,:1],_fY),axis=2)

    _fooledY = np.reshape(_fooledY, [params["image_dims"][0], params["image_dims"][1], 2])
    _fooledY = np.concatenate((v[:,:,:1],_fooledY),axis=2)

    print(np.max(np.array(_fY)))

    # Convert to RGB to get plots
    v = ycbcr2rgb(np.array([v])*255)[0]/255.
    _w = ycbcr2rgb(np.array([_w])*255)[0]/255.
    _fY = ycbcr2rgb(np.array([_fY])*255)[0]/255.
    _fooledY = ycbcr2rgb(np.array([_fooledY])*255)[0]/255.

    # print(np.max(np.abs(_fY)))
    # print(np.max(np.abs(_v)))

    X_org = np.array([[v]])
    X_new = np.array([[_w]])
    fY = np.array([[_fY]])
    fooledY = np.array([[_fooledY]])

    return X_org, X_new, fY, fooledY

def adjust_images(X_org, X_new, fY, fooledY, size=200):
    """
    Used to adjust images for super-resolution
    """
    # Adjust maximum and minimum value
    X_org_ = np.array([x if x <= 1.0 else 1.0 for x in X_org.flatten()])
    X_org_ = np.array([x if x >= 0.0 else 0.0 for x in X_org_.flatten()])

    X_new_ = np.array([x if x <= 1.0 else 1.0 for x in X_new.flatten()])
    X_new_ = np.array([x if x >= 0.0 else 0.0 for x in X_new_.flatten()])

    fY_ = np.array([x if x <= 1.0 else 1.0 for x in fY.flatten()])
    fY_ = np.array([x if x >= 0.0 else 0.0 for x in fY_.flatten()])

    fooledY_ = np.array([x if x <= 1.0 else 1.0 for x in fooledY.flatten()])
    fooledY_ = np.array([x if x >= 0.0 else 0.0 for x in fooledY_.flatten()])

    # Convert to basic grayscale numpy arrays
    X_org = X_org_.reshape([X_org.shape[2], X_org.shape[3]])
    X_new = X_new_.reshape([X_new.shape[2], X_new.shape[3]])

    fY = fY_.reshape([fY.shape[2], fY.shape[3], 4])
    fooledY = fooledY_.reshape([fooledY.shape[2], fooledY.shape[3], 4])

    fY = reshape_conversion(fY)
    fooledY = reshape_conversion(fooledY)

    # Convert to equal sizes
    X_org = convert_image(X_org, size)
    X_new = convert_image(X_new, size)
    fY = convert_image(fY, size)
    fooledY = convert_image(fooledY, size)

    # Reshape to plot using helper function
    X_org = X_org.reshape([1,1,size,size,1])
    X_new = X_new.reshape([1,1,size,size,1])
    fY = fY.reshape([1,1,size,size,1])
    fooledY = fooledY.reshape([1,1,size,size,1])

    return X_org, X_new, fY, fooledY

"""
Resizing and import for superresolution
"""
def set_image_alignment(image, alignment):
	alignment = int(alignment)
	width, height = image.shape[1], image.shape[0]
	width = (width // alignment) * alignment
	height = (height // alignment) * alignment

	if image.shape[1] != width or image.shape[0] != height:
		image = image[:height, :width, :]

	if len(image.shape) >= 3 and image.shape[2] >= 4:
		image = image[:, :, 0:3]
	return image

def resize_image_by_pil(image, scale, resampling_method="bicubic"):
	width, height = image.shape[1], image.shape[0]
	new_width = int(width * scale)
	new_height = int(height * scale)

	if resampling_method == "bicubic":
		method = Image.BICUBIC
	elif resampling_method == "bilinear":
		method = Image.BILINEAR
	elif resampling_method == "nearest":
		method = Image.NEAREST
	else:
		method = Image.LANCZOS

	if len(image.shape) == 3 and image.shape[2] == 3:
		image = Image.fromarray(image, "RGB")
		image = image.resize([new_width, new_height], resample=method)
		image = np.asarray(image)
	elif len(image.shape) == 3 and image.shape[2] == 4:
		# the image may has an alpha channel
		image = Image.fromarray(image, "RGB")
		image = image.resize([new_width, new_height], resample=method)
		image = np.asarray(image)
	else:
		image = Image.fromarray(image.reshape(height, width))
		image = image.resize([new_width, new_height], resample=method)
		image = np.asarray(image)
		image = image.reshape(new_height, new_width, 1)
	return image

def load_image(filename, width=0, height=0, channels=0, alignment=0, print_console=True):
	if not os.path.isfile(filename):
		raise LoadError("File not found [%s]" % filename)
	image = misc.imread(filename)

	if len(image.shape) == 2:
		image = image.reshape(image.shape[0], image.shape[1], 1)
	if (width != 0 and image.shape[1] != width) or (height != 0 and image.shape[0] != height):
		raise LoadError("Attributes mismatch")
	if channels != 0 and image.shape[2] != channels:
		raise LoadError("Attributes mismatch")
	if alignment != 0 and ((width % alignment) != 0 or (height % alignment) != 0):
		raise LoadError("Attributes mismatch")

	if print_console:
		print("Loaded [%s]: %d x %d x %d" % (filename, image.shape[1], image.shape[0], image.shape[2]))
	return image

def build_input_image(image, width=0, height=0, channels=1, scale=1, alignment=0, convert_ycbcr=True, jpeg_mode=False):
    """
    build input image from file.
    crop, adjust the image alignment for the scale factor, resize, convert color space.
    """
    if width != 0 and height != 0:
        if image.shape[0] != height or image.shape[1] != width:
            x = (image.shape[1] - width) // 2
            y = (image.shape[0] - height) // 2
            image = image[y: y + height, x: x + width, :]

    if image.shape[2] >= 4:
        image = image[:, :, 0:3]

    if alignment > 1:
        image = set_image_alignment(image, alignment)

    if scale != 1:
        image = resize_image_by_pil(image, 1.0 / scale)

    if channels == 1 and image.shape[2] == 3:
        if convert_ycbcr:
            image = rgb2y([image])
            print(np.array(image).shape)

    return image

def load_input_image(filename, width=0, height=0, channels=1, scale=1, alignment=0, convert_ycbcr=True,
                     jpeg_mode=False, print_console=True):
	image = load_image(filename, print_console=print_console)
	return build_input_image(image, width, height, channels, scale, alignment, convert_ycbcr, jpeg_mode)

def reshape_conversion(original_pixels):

    original_width = original_pixels.shape[0]
    original_height = original_pixels.shape[1]
    original_conv = 2

    pixels = np.zeros((original_conv*original_width, original_conv*original_height))

    for x in range(original_width):
        for y in range(original_height):

            pixels[original_conv*x][original_conv*y] = original_pixels[x][y][0]
            pixels[original_conv*x+1][original_conv*y] = original_pixels[x][y][1]
            pixels[original_conv*x][original_conv*y+1] = original_pixels[x][y][2]
            pixels[original_conv*x+1][original_conv*y+1] = original_pixels[x][y][3]

    return pixels
"""
Figures and images
"""
def save_psnr_fig(X, fname, adv_dict, legend=True, format="eps"):
    """
    Save psnr figure (input vs output psnr)
    """
    epsilon = X[:,adv_dict['psnr_input']]
    fig = plt.figure()
    ax = plt.subplot(111)

    style = ['-ro', '--bx', '-g*', '-ms', '-^k', '-go', '-^r', '--g', '-^b', '--r', '--k', '-ro', '--bx', '-g*', '-ms', '-^k', '-go', '-^b', '--r', '--k', '-ro', '--bx', '-g*', '-ms', '-^k', '-go']

    legend_strs = [k for k, v in adv_dict.items()]
    legend_strs = legend_strs[1:len(legend_strs)]
    print(legend_strs)

    used_legend = []
    for i in range(len(legend_strs)):
        if sum(X[:, adv_dict[legend_strs[i]]]) != 0.0:
            ax.plot(epsilon, X[:, adv_dict[legend_strs[i]]], style[i], markersize=5)
            used_legend.append(legend_strs[i])

    plt.xlabel('input PSNR (dB)', fontsize=12)
    plt.ylabel('output PSNR (dB)', fontsize=12)
    ax.grid()
    # plt.ylim((-5,105))

    if legend==True:
        ax.legend(used_legend, fontsize=10)

    ax.tick_params(axis='both', labelsize='large')
    plt.savefig(fname + "." + format, format=format, dpi=500)
    plt.close(fig)

def save_psnr_fig_2(X, fname, adv_dict, legend=True, format="eps"):
    """
    Save psnr figure (input epsilon vs output psnr)
    """
    epsilon = X[:,adv_dict['psnr_input']]
    fig = plt.figure()
    ax = plt.subplot(111)

    style = ['-ro', '--bx', '-g*', '-ms', '-^k', '-go', '-^r', '--g', '-^b', '--r', '--k', '-ro', '--bx', '-g*', '-ms', '-^k', '-go', '-^b', '--r', '--k', '-ro', '--bx', '-g*', '-ms', '-^k', '-go']

    legend_strs = [k for k, v in adv_dict.items()]
    legend_strs = legend_strs[1:len(legend_strs)]
    print(legend_strs)

    for i in range(len(legend_strs)):
        if sum(X[:, adv_dict[legend_strs[i]]]) != 0.0:
            ax.plot(epsilon, X[:, adv_dict[legend_strs[i]]], style[i], markersize=5)
            print(legend_strs[i])

    plt.xlabel('epsilon', fontsize=12)
    plt.ylabel('output PSNR (dB)', fontsize=12)
    ax.grid()
    # plt.ylim((-5,105))
    if legend==True:
        ax.legend(legend_strs, fontsize=10)

    ax.tick_params(axis='both', labelsize='large')
    plt.savefig(fname + "." + format, format=format, dpi=500)
    plt.close(fig)

def save_image_fig(img_mtx, params, epsilon_range, names, title, img_size = [32,32,1], format="eps"):
    """
    Save adversarial examples in compact plot

    Input
        : img_mtx (dictionary for each image set)
        : names (img_mtx dictionary keys to be plotted)

    Output
        : {title}_{epsilon}.jpg (images)
    """

    print(img_size)

    # Layout
    cols = (2 * len(names)) + 2
    rows = 10

    for idx in range(len(epsilon_range)):
        # Settings
        fig = plt.figure(figsize=(cols, rows))
        plt.tight_layout()

        # Original
        images = np.array(img_mtx[names[0]])
        images = images.flatten()
        for i, x in enumerate(images):
            if x < 1 and x > 0:
                images[i] = x
            elif x >= 1:
                images[i] = 1
            else:
                images[i] = 0
        images = images.reshape(len(epsilon_range), 10, 4, img_size[0], img_size[1], img_size[2])

        color_map = params['colors_input']
        if color_map == 'rgb' or color_map == 'y':
            color_map = None

        for batch_idx in range(images.shape[1]):

            fX = images[:,batch_idx,0,:,:]
            fY = images[:,batch_idx,2,:,:]

            if params["colors_input"] == "y":
                rgb_images = images[:,batch_idx,0,:,:]
                rgb_image = rgb_images[idx]
                fX = rgb2ycbcr(np.array([rgb_image*255.]))/255.
                fX = fX[0,:,:,0]

                fig.add_subplot(rows, cols, 1 + (batch_idx * cols))
                plt.imshow(np.squeeze(fX), cmap="gray")
                plt.axis('off')
                plt.title("input\norg", fontsize=9) if batch_idx == 0 else None
            else:
                fig.add_subplot(rows, cols, 1 + (batch_idx * cols))
                plt.imshow(np.squeeze(fX[idx]), cmap=color_map)
                plt.axis('off')
                plt.title("input\norg", fontsize=9) if batch_idx == 0 else None

            fig.add_subplot(rows, cols, (2 + len(names)) + (batch_idx * cols))
            plt.imshow(np.squeeze(fY[idx]), cmap=color_map)
            plt.axis('off')
            plt.title("output\norg", fontsize=9) if batch_idx == 0 else None

        color_map = params['colors_output']
        if color_map == 'rgb' or color_map == 'cbcr':
            color_map = None

        for name_idx in range(len(names)):
            # Get image
            images = np.array(img_mtx[names[name_idx]])
            images = images.flatten()
            for i, x in enumerate(images):
                if x < 1 and x > 0:
                    images[i] = x
                elif x >= 1:
                    images[i] = 1
                else:
                    images[i] = 0
            images = images.reshape(len(epsilon_range), 10, 4, img_size[0], img_size[1], img_size[2])

            for batch_idx in range(images.shape[1]):
                fX = images[:,batch_idx,1,:,:]
                fY = images[:,batch_idx,3,:,:]

                if params['colors_output'] == 'ycbr':
                    fX = convert_ycbcr_to_rgb(fX)[idx]
                    fY = convert_rgb_to_ycbcr(fY)

                if params["colors_input"] == "y":
                    fX = rgb2ycbcr(np.array([fX[idx]*255.]))/255.
                    fX = fX[0,:,:,0]

                    fig.add_subplot(rows, cols, (2 + name_idx) + (batch_idx * cols))
                    plt.imshow(np.squeeze(fX), cmap="gray")
                    plt.axis('off')
                    plt.title("input\n{}".format(names[name_idx]), fontsize=9) if batch_idx == 0 else None
                else:
                    fig.add_subplot(rows, cols, (2 + name_idx) + (batch_idx * cols))
                    plt.imshow(np.squeeze(fX[idx]), cmap=color_map)
                    plt.axis('off')
                    plt.title("input\n{}".format(names[name_idx]), fontsize=9) if batch_idx == 0 else None

                fig.add_subplot(rows, cols, (2 + name_idx + (len(names) + 1)) + (batch_idx * cols))
                plt.imshow(np.squeeze(fY[idx]), cmap=color_map)
                plt.axis('off')
                plt.title("output\n{}".format(names[name_idx]), fontsize=9) if batch_idx == 0 else None

        plt.savefig("./results/images/compact/{}_{}_{:.2f}.{}".format(params["model"], title, float(epsilon_range[idx]), format), format=format, dpi=500)
        plt.close(fig)
