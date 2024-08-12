import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
import scipy

img_index = 1

# 3.1
def read_im_2npyArray(path):
	img = plt.imread(path)

	return np.asarray(img)

# 3.2
def color_map(keyColors, cmap_name, N=256):

	return LinearSegmentedColormap.from_list(cmap_name, keyColors, N)

# 3.3
def show_color_map(img, cmap):
	display_image(img, cmap)

# 3.4
def split_rgb(img_array):

	return img_array[:,:,0], img_array[:,:,1], img_array[:,:,2]

# 3.5
def combine_comps(r, g, b):
	rgb = np.dstack((r, g, b))

	return rgb

# 3.6
def display_image(img, cmap=None):
	if cmap is not None:
		plt.imshow(img, cmap)
	else:
		plt.imshow(img)

	return

# 5.1
def rgb_to_ycbcr(img_array, verbose=False, visualize=False):
	# Conversion matrix
	aux_mat = np.array([[0.299, 0.587, 0.114],
						[-0.168736, -0.331264, 0.5],
						[0.5, -0.418688, -0.081312]])

	# Convert from RGB to YCbCr
	ycbcr = np.empty_like(img_array).astype(np.float64)
	ycbcr[:,:,0] = np.dot(img_array, aux_mat.T[:,0]) # Y
	ycbcr[:,:,1] = 128 + np.dot(img_array, aux_mat.T[:,1]) # Cb
	ycbcr[:,:,2] = 128 + np.dot(img_array, aux_mat.T[:,2]) # Cr

	if visualize:
		global img_index
		fig = plt.figure()
		fig.canvas.manager.set_window_title(f'{img_index}: RGB to YCbCr - Encoder')
		img_index += 1

		plt.subplot(1, 3, 1)
		display_image(ycbcr[:,:,0], cmap='gray')
		plt.title('Y')
		plt.subplot(1, 3, 2)
		display_image(ycbcr[:,:,1], cmap='gray')
		plt.title('Cb')
		plt.subplot(1, 3, 3)
		display_image(ycbcr[:,:,2], cmap='gray')
		plt.title('Cr')

	return ycbcr[:,:,0], ycbcr[:,:,1], ycbcr[:,:,2]

# 5.2
def ycbcr_to_rgb(ycbcr_im, visualize=False):

	ycbcr_im = ycbcr_im.astype(np.float64)
	ycbcr_im[:, :,  1] -=  128
	ycbcr_im[:, :,  2] -=  128

	# Define the inverse conversion matrix
	inv_aux_mat = np.linalg.inv(
						np.array([[0.299, 0.587, 0.114],
						[-0.168736, -0.331264, 0.5],
						[0.5, -0.418688, -0.081312]])
				)

	# Convert from YCbCr to RGB
	rgb = np.round(np.dot(ycbcr_im, inv_aux_mat.T))

	# Clip the values to the range [0, 255] and convert to float64
	rgb = np.clip(rgb,  0,  255)

	rgb = rgb.astype(np.uint8)

	if visualize:
		plt.figure()
		display_image(rgb)

	return rgb

# 4.1
def padding_encode(image_array):
	height, width = image_array.shape

	if height % 32 == 0 and width % 32 == 0:
		return image_array

	pad_height = 32 - height % 32
	pad_width = 32 - width % 32

	# Applies padding
	last_line = image_array[height - 1:height, :]
	image_array = np.vstack((image_array, np.repeat(last_line, pad_height, axis=0)))

	last_column = image_array[:, width - 1:width]
	image_array = np.hstack((image_array, np.repeat(last_column, pad_width, axis=1)))

	return image_array

# 4.2
def padding_decode(height, width, pad_encoded_img):

	return pad_encoded_img[:height, :width]

# 6.1
def downsampling(Y, Cb, Cr, Cb_compression, Cr_compression, interpolation = cv2.INTER_LINEAR, visualize=False):
	Y_d = Y
	if Cr_compression != 0:
		Cb_d = cv2.resize(Cb, (0,0), fx = Cb_compression * 0.25, fy = 1, interpolation=interpolation)
		Cr_d = cv2.resize(Cr, (0,0), fx = Cr_compression * 0.25, fy = 1, interpolation=interpolation)

	else:
		Cb_d = cv2.resize(Cb, (0,0), fx = Cb_compression * 0.25, fy = Cb_compression * 0.25, interpolation=interpolation)
		Cr_d = cv2.resize(Cr, (0,0), fx = Cb_compression * 0.25, fy = Cb_compression * 0.25, interpolation=interpolation)

	if visualize:
		global img_index
		fig = plt.figure()
		fig.canvas.manager.set_window_title(f'{img_index}: Downsampling - Encoder')
		img_index += 1

		plt.subplot(1, 3, 1)
		display_image(Y, cmap='gray')
		plt.title('Y downsampling 4:' + str(Cb_compression) + ":" + str(Cr_compression))
		plt.subplot(1, 3, 2)
		display_image(Cb_d, cmap='gray')
		plt.title('Cb_d downsampling 4:' + str(Cb_compression) + ":" + str(Cr_compression))
		plt.subplot(1, 3, 3)
		display_image(Cr_d, cmap='gray')
		plt.title('Cr_d downsampling 4:' + str(Cb_compression) + ":" + str(Cr_compression))
	
	return Y_d, Cb_d, Cr_d

# 6.2
def upsampling(Y, Cb, Cr, Cb_decompression, Cr_decompression, interpolation = cv2.INTER_LINEAR, visualize=False):
	Y_d = Y

	if Cr_decompression != 0:
		Cb_d = cv2.resize(Cb, (0,0), fx = 4 / Cb_decompression, fy = 1, interpolation=interpolation)
		Cr_d = cv2.resize(Cr, (0,0), fx = 4 / Cr_decompression, fy = 1, interpolation=interpolation)

	else:
		Cb_d = cv2.resize(Cb, (0,0), fx = 4 / Cb_decompression, fy = 4 / Cb_decompression, interpolation=interpolation)
		Cr_d = cv2.resize(Cr, (0,0), fx = 4 / Cb_decompression, fy = 4 / Cb_decompression, interpolation=interpolation)
	if visualize:
		plt.subplot(1, 3, 1)
		display_image(Y, cmap='gray')
		plt.title('Y upsampling 4:' + str(Cb_decompression) + ":" + str(Cr_decompression))
		plt.subplot(1, 3, 2)
		display_image(Cb_d, cmap='gray')
		plt.title('Cb_d upsampling 4:' + str(Cb_decompression) + ":" + str(Cr_decompression))
		plt.subplot(1, 3, 3)
		display_image(Cr_d, cmap='gray')
		plt.title('Cr_d upsampling 4:' + str(Cb_decompression) + ":" + str(Cr_decompression))

	return Y_d, Cb_d, Cr_d

# 7.1.1
def full_channel_dct(full_channel):
	# axis=0, to affect the rows
	full_channel = full_channel.astype(np.float64)
	return scipy.fftpack.dct(scipy.fftpack.dct(full_channel, axis=0, norm='ortho').T, axis=0, norm='ortho').T

# 7.1.2
def full_channel_idct(dct_full_channel):
	# axis=0, to affect the rows
	return scipy.fftpack.idct(scipy.fftpack.idct(dct_full_channel, axis=0, norm='ortho').T, axis=0, norm='ortho').T

# 7.1.3
def visualize_dct(Y_dct, Cb_dct, Cr_dct, title="", y_title="Y_dct", cb_title="Cb_dct", cr_title="Cr_dct"):
	fig = plt.figure(figsize=(10, 10))
	global img_index
	fig.canvas.manager.set_window_title(f'{img_index}: {title}')
	img_index += 1

	plt.subplot(131)
	display_image(np.log(np.abs(Y_dct) + 0.0001), cmap='gray')
	plt.title(y_title)
	plt.axis('off')

	plt.subplot(132)
	display_image(np.log(np.abs(Cb_dct) + 0.0001), cmap='gray')
	plt.title(cb_title)
	plt.axis('off')

	plt.subplot(133)
	display_image(np.log(np.abs(Cr_dct) + 0.0001), cmap='gray')
	plt.title(cr_title)
	plt.axis('off')

# 7.2.1
def dct_by_blocks(full_channel, block_size):
	aux_mat = np.zeros_like(full_channel, dtype=np.float64)

	for i in range(0, full_channel.shape[0], block_size):
		for j in range(0, full_channel.shape[1], block_size):
			aux_mat[i:i + block_size, j:j + block_size] = full_channel_dct(full_channel[i:i + block_size, j:j + block_size])

	return aux_mat

# 7.2.2
def idct_by_blocks(dct_full_channel, block_size):
	aux_mat = np.zeros_like(dct_full_channel, dtype=np.float64)

	for i in range(0, dct_full_channel.shape[0], block_size):
		for j in range(0, dct_full_channel.shape[1], block_size):
			aux_mat[i:i + block_size, j:j + block_size] = full_channel_idct(dct_full_channel[i:i + block_size, j:j + block_size])

	return aux_mat

q_y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
			  [12, 12, 14, 19, 26, 58, 60, 55],
			  [14, 13, 16, 24, 40, 57, 69, 56],
			  [14, 17, 22, 29, 51, 87, 80, 62],
			  [18, 22, 37, 56, 68, 109, 103, 77],
			  [24, 35, 55, 64, 81, 104, 113, 92],
			  [49, 64, 78, 87, 103, 121, 120, 101],
			  [72, 92, 95, 98, 112, 100, 103, 99]]
			  )

q_cbcr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
				  [18, 21, 26, 66, 99, 99, 99, 99],
				  [24, 26, 56, 99, 99, 99, 99, 99],
				  [47, 66, 99, 99, 99, 99, 99, 99],
				  [99, 99, 99, 99, 99, 99, 99, 99],
				  [99, 99, 99, 99, 99, 99, 99, 99],
				  [99, 99, 99, 99, 99, 99, 99, 99],
				  [99, 99, 99, 99, 99, 99, 99, 99]]
				  )

# 8.1
def calculate_sf_factor(qf):
	if qf < 50:
		sf = 50 / qf
	else:
		sf = 2 - qf / 50
	
	return sf

def get_quantization_matrix(qf, is_Y=True):
	sf = calculate_sf_factor(qf)

	if sf != 0: 																#? Qs = clip( round(q * sf) , 1, 255)
		return np.clip(np.round((q_y if is_Y else q_cbcr) * sf), 1, 255).astype(np.uint8)
	else:
		return None

def block_quantization(x_dct_block, quantization_matrix):
    if quantization_matrix is not None:
        x_Q = np.round(x_dct_block / quantization_matrix.astype(np.float64))
    else:
        x_Q = np.round(x_dct_block)

    return x_Q

def block_dequantization(x_Q_block, quantization_matrix, verbose=False):
    if quantization_matrix is not None:
        x_dct_block = x_Q_block * quantization_matrix.astype(np.float64)
    else:
        x_dct_block = x_Q_block
    if verbose:
        print("Dequantized DCT block matrix: ")
        print(x_dct_block)
    return x_dct_block

def quantization(x_dct, qf, is_Y=True, verbose=False):
    x_Q = np.zeros_like(x_dct, dtype=np.float64)
    quantization_matrix = get_quantization_matrix(qf, is_Y)
    for i in range(0, x_dct.shape[0], 8):
        for j in range(0, x_dct.shape[1], 8):
            x_Q[i:i + 8, j:j + 8] = block_quantization(x_dct[i:i + 8, j:j + 8], quantization_matrix)
    return x_Q

def dequantization(x_dct, qf, is_Y=True, verbose=False):
    x_Q = np.zeros_like(x_dct, dtype=np.float64)
    quantization_matrix = get_quantization_matrix(qf, is_Y)
    for i in range(0, x_dct.shape[0], 8):
        for j in range(0, x_dct.shape[1], 8):
            x_Q[i:i + 8, j:j + 8] = block_dequantization(x_dct[i:i + 8, j:j + 8], quantization_matrix, verbose=verbose)
    return x_Q

# 9.1 DIFF = DCi - DCi-1
def dc_dpcm_coding(Y, Cb, Cr):
	dpcm = [Y, Cb, Cr]

	for n in range(3):															# Iterate over y, cb, cr
		temp = 0
		last_dc = 0
		for i in range(0, np.shape(dpcm[n])[0], 8):								# Rows
			for j in range(0, np.shape(dpcm[n])[1], 8): 						# Columns
				temp = dpcm[n][i, j]
				dpcm[n][i, j] = dpcm[n][i, j] - last_dc
				last_dc = temp

	return dpcm[0], dpcm[1], dpcm[2]

# 9.2 DCi = DCi-1 + DIFF
def dc_dpcm_decoding(Y, Cb, Cr):
	dpcm = [Y, Cb, Cr]

	for n in range(3):															# Iterate over y, cb, cr
		last_dc = 0
		for i in range(0, np.shape(dpcm[n])[0], 8):								# Rows
			for j in range(0, np.shape(dpcm[n])[1], 8): 						# Columns
				last_dc = dpcm[n][i, j] = dpcm[n][i, j] + last_dc

	return dpcm[0], dpcm[1], dpcm[2]


# 2
# ENCODING FUNCTION
'''
path:				Path to the image to encode
verbose:			True to show relevant information (like entropy)
visualize:			True to display images
dct_blocks:			Blocks to apply the dct, None to apply without any window
interpolation:		Downsampling interpolation
cb_down_samp:		Cb's channel downsampling value
cr_down_samp:		Cr's channel downsampling value
save_path:			Path to save relevant images, if None doesn't save

'''

def encoder(path, qf=50, verbose=False, visualize=False, dct_blocks=None, interpolation=cv2.INTER_LINEAR, cb_down_samp=2, cr_down_samp=2, save_path=None):
	
	if verbose:
		print("Opening the image at: " + path)

	img_array = read_im_2npyArray(path) 										# Convert to numpy array

	height, width, _ = img_array.shape 											# _ to ignore 3rd shape/array

	if verbose:
		print("Splitting the image into its RGB components")
	r, g, b = split_rgb(img_array) 												# Split the image into its RGB components
	

	cm_r = color_map([(0,0,0), (1,0,0)],'Red')
	cm_g = color_map([(0,0,0), (0,1,0)],'Green')
	cm_b = color_map([(0,0,0), (0,0,1)],'Blue')

	if visualize:
		fig = plt.figure()
		global img_index
		fig.canvas.manager.set_window_title(f'{img_index}: RGB split - Encoder')
		img_index += 1

		# Display the original image
		plt.subplot(2, 3, 1)
		display_image(img_array)
		plt.title('Original')

		# Display the sum of the channels
		plt.subplot(2, 3, 3)
		display_image(combine_comps(r, g, b))
		plt.title('R + G + B')

		# Display the separated channels		
		plt.subplot(2, 3, 4)
		show_color_map(r, cm_r)
		plt.title('Red channel')

		plt.subplot(2, 3, 5)
		show_color_map(g, cm_g)
		plt.title('Green channel')

		plt.subplot(2, 3, 6)
		show_color_map(b, cm_b)
		plt.title('Blue channel')

	if save_path is not None:
		plt.imsave(save_path + 'R_enc.bmp', r, cmap=cm_r)
		plt.imsave(save_path + 'G_enc.bmp', g, cmap=cm_g)
		plt.imsave(save_path + 'B_enc.bmp', b, cmap=cm_b)

	r = padding_encode(r)
	g = padding_encode(g)
	b = padding_encode(b)

	# r, g, b int
	if verbose:
		print("Image split completed")
		print("red:\n")
		print(r)
		print("green:\n")
		print(g)
		print("blue:\n")
		print(b)

		print("Is the original image equal to the sum of R+G+B? " + str(np.array_equal((r + g + b), img_array)))

	if visualize:
		fig = plt.figure()
		fig.canvas.manager.set_window_title(f'{img_index}: Padding - Encoder')
		img_index += 1

		# Display the original image
		plt.subplot(2, 3, 1)
		display_image(img_array)
		plt.title('Original')

		# Display the sum of the channels
		plt.subplot(2, 3, 3)
		display_image(combine_comps(r, g, b))
		plt.title('R + G + B')

		# Display the separated channels		
		plt.subplot(2, 3, 4)
		cm = color_map([(0,0,0), (1,0,0)],'Red')
		show_color_map(r, cm)
		plt.title('Red channel')

		plt.subplot(2, 3, 5)
		cm = color_map([(0,0,0), (0,1,0)],'Green')
		show_color_map(g, cm)
		plt.title('Green channel')

		plt.subplot(2, 3, 6)
		cm = color_map([(0,0,0), (0,0,1)],'Blue')
		show_color_map(b, cm)
		plt.title('Blue channel')
	
	# 5.3.1 & 5.3.2
	Y, Cb, Cr = rgb_to_ycbcr(combine_comps(r, g, b), visualize=visualize)
	if save_path is not None:
		plt.imsave(save_path + 'Y_enc.bmp', Y, cmap='gray')
		plt.imsave(save_path + 'Cb_enc.bmp', Cb, cmap='gray')
		plt.imsave(save_path + 'Cr_enc.bmp', Cr, cmap='gray')

	# 6.3
	Y_d, Cb_d, Cr_d = downsampling(Y, Cb, Cr, cb_down_samp, cr_down_samp, interpolation=interpolation, visualize=visualize)
	if save_path is not None:
		plt.imsave(save_path + 'Y_dwn_smp_4_'+ str(cb_down_samp) + "_" + str(cr_down_samp) + '_enc.bmp', Y_d, cmap='gray')
		plt.imsave(save_path + 'Cb_dwn_smp_4_'+ str(cb_down_samp) + "_" + str(cr_down_samp) + '_enc.bmp', Cb_d, cmap='gray')
		plt.imsave(save_path + 'Cr_dwn_smp_4_'+ str(cb_down_samp) + "_" + str(cr_down_samp) + '_enc.bmp', Cr_d, cmap='gray')

	# 7.2.3
	if dct_blocks is not None:
		Y_dct = dct_by_blocks(Y_d, dct_blocks)
		Cb_dct = dct_by_blocks(Cb_d, dct_blocks) 
		Cr_dct = dct_by_blocks(Cr_d, dct_blocks)

	else:
		# 7.1.1
		Y_dct = full_channel_dct(Y_d)
		Cb_dct = full_channel_dct(Cb_d)
		Cr_dct = full_channel_dct(Cr_d)

	# 7.1.3
	if visualize:
		visualize_dct(Y_dct, Cb_dct, Cr_dct, "DCT - Encoder")

	block_str = "full" if dct_blocks is None else str(dct_blocks) + "x" + str(dct_blocks)
	if save_path is not None:
		plt.imsave(save_path + 'Y_dct_' + block_str + '_enc.bmp', np.log(np.abs(Y_dct) + 0.0001), cmap='gray')
		plt.imsave(save_path + 'Cb_dct_' + block_str + '_enc.bmp', np.log(np.abs(Cb_dct) + 0.0001), cmap='gray')
		plt.imsave(save_path + 'Cr_dct_' + block_str + '_enc.bmp', np.log(np.abs(Cr_dct) + 0.0001), cmap='gray')

	y_Q = quantization(Y_dct, qf, is_Y=True, verbose=verbose)
	Cb_Q = quantization(Cb_dct, qf, is_Y=False, verbose=verbose)
	Cr_Q = quantization(Cr_dct, qf, is_Y=False, verbose=verbose)

	if save_path is not None:
		plt.imsave(save_path + 'Y_qnt_qf_' + str(qf) + '_enc.bmp', np.log(np.abs(y_Q) + 0.0001), cmap='gray')
		plt.imsave(save_path + 'Cb_qnt_qf_' + str(qf) + '_enc.bmp', np.log(np.abs(Cb_Q) + 0.0001), cmap='gray')
		plt.imsave(save_path + 'Cr_qnt_qf_' + str(qf) + '_enc.bmp', np.log(np.abs(Cr_Q) + 0.0001), cmap='gray')

	## 9.3
	if verbose:
		print('Entropy before dpcm:')
		print(f'y_Q: {entropy(y_Q)}')
		print(f'Cb_Q: {entropy(Cb_Q)}')
		print(f'Cb_Q: {entropy(Cr_Q)}')
	Y_dpcm, Cb_dpcm, Cr_dpcm = dc_dpcm_coding(y_Q, Cb_Q, Cr_Q)
	if verbose:
		print('Entropy after dpcm:')
		print(f'y_Q: {entropy(Y_dpcm)}')
		print(f'Cb_Q: {entropy(Cb_dpcm)}')
		print(f'Cb_Q: {entropy(Cr_dpcm)}')

	if save_path is not None:
		plt.imsave(save_path + 'Y_dpcm_qf_' + str(qf) + '_enc.bmp', np.log(np.abs(y_Q) + 0.0001), cmap='gray')
		plt.imsave(save_path + 'Cb_dpcm_qf_' + str(qf) + '_enc.bmp', np.log(np.abs(Cb_Q) + 0.0001), cmap='gray')
		plt.imsave(save_path + 'Cr_dpcm_qf_' + str(qf) + '_enc.bmp', np.log(np.abs(Cr_Q) + 0.0001), cmap='gray')

	if visualize:
		visualize_dct(y_Q, Cb_Q, Cr_Q, "Quantization - Encoder", "Y_Q", "Cb_Q", "Cr_Q")
		visualize_dct(Y_dpcm, Cb_dpcm, Cr_dpcm, "DPCM - Encoder", "Y_dpcm", "Cb_dpcm", "Cr_dpcm")

	return Y_dpcm, Cb_dpcm, Cr_dpcm, height, width, Y

# DECODING FUNCTION
'''
verbose:			True to show relevant information (like entropy)
visualize:			True to display images
dct_blocks:			Blocks to apply the dct, None to apply without any window
interpolation:		Downsampling interpolation
cb_down_samp:		Cb's channel downsampling value
cr_down_samp:		Cr's channel downsampling value
save_path:			Path to save relevant images, if None doesn't save
'''

def decoder(Y_dpcm, Cb_dpcm, Cr_dpcm, height, width, qf=50, verbose=False, visualize=False, dct_blocks=None, interpolation=cv2.INTER_LINEAR, cb_down_samp=2, cr_down_samp=2, save_path=None):
	global img_index
	
	# 9.4
	Y_q, Cb_q , Cr_q = dc_dpcm_decoding(Y_dpcm, Cb_dpcm, Cr_dpcm)

	Y_dQ = dequantization(Y_q, qf, is_Y=True, verbose=verbose)
	Cb_dQ = dequantization(Cb_q, qf, is_Y=False, verbose=verbose)
	Cr_dQ = dequantization(Cr_q, qf, is_Y=False, verbose=verbose)

	if dct_blocks is not None:
		# 7.2.4
		Y_d = idct_by_blocks(Y_dQ, dct_blocks)
		Cb_d = idct_by_blocks(Cb_dQ, dct_blocks)
		Cr_d = idct_by_blocks(Cr_dQ, dct_blocks)
	else:
		# 7.1.4
		Y_d = full_channel_idct(Y_dQ)
		Cb_d = full_channel_idct(Cb_dQ)
		Cr_d = full_channel_idct(Cr_dQ)

	# Y_d, Cb_d, Cr_d float64
	if visualize:
		fig = plt.figure()
		fig.canvas.manager.set_window_title(f'{img_index}: iDCT - Decoder')
		img_index += 1

		plt.subplot(1, 3, 1)
		display_image(Y_d, cmap='gray')
		plt.title('Y')
		plt.subplot(1, 3, 2)
		display_image(Cb_d, cmap='gray')
		plt.title('Cb')
		plt.subplot(1, 3, 3)
		display_image(Cr_d, cmap='gray')
		plt.title('Cr')

	# 6.4
	Y, Cb, Cr = upsampling(Y_d, Cb_d, Cr_d, cb_down_samp, cr_down_samp, interpolation=interpolation, visualize=False)
	# Y, Cb, Cr float64

	if visualize:
		fig = plt.figure()
		fig.canvas.manager.set_window_title(f'{img_index}: Upsampling - Decoder')
		img_index += 1

		plt.subplot(1, 3, 1)
		display_image(Y[:,:], cmap='gray')
		plt.title('Y')
		plt.subplot(1, 3, 2)
		display_image(Cb[:,:], cmap='gray')
		plt.title('Cb')
		plt.subplot(1, 3, 3)
		display_image(Cr[:,:], cmap='gray')
		plt.title('Cr')

	total = combine_comps(Y, Cb, Cr) # RGB components
	total = ycbcr_to_rgb(total)

	if verbose:
		print("First pixel of the RGB image: " + str(total[0][0]))

	if visualize:
		fig = plt.figure()
		fig.canvas.manager.set_window_title(f'{img_index}: YCbCr to RGB - Decoder')
		img_index += 1

		# Display the original image
		plt.subplot(2, 2, 1)
		display_image(total)
		plt.title('Decoded')

		# Display the separated channels
		plt.subplot(2, 2, 2)
		cm = color_map([(0,0,0), (1,0,0)],'Red')
		show_color_map(total[:,:,0], cm)
		plt.title('Red channel')

		plt.subplot(2, 2, 3)
		cm = color_map([(0,0,0), (0,1,0)],'Green')
		show_color_map(total[:,:,1], cm)
		plt.title('Green channel')

		plt.subplot(2, 2, 4)
		cm = color_map([(0,0,0), (0,0,1)],'Blue')
		show_color_map(total[:,:,2], cm)
		plt.title('Blue channel')

	# 4.2 Undo padding
	R = padding_decode(height, width, total[:,:,0])
	G = padding_decode(height, width, total[:,:,1])
	B = padding_decode(height, width, total[:,:,2])
	
	total = combine_comps(R, G, B)
	
	if visualize:
		fig = plt.figure()
		fig.canvas.manager.set_window_title(f'{img_index}: RGB split: Depadding - Decoder')
		img_index += 1

		# Display the original image
		plt.subplot(2, 2, 1)
		display_image(total)
		plt.title('Decoded')

		# Display the separated channels
		plt.subplot(2, 2, 2)
		cm = color_map([(0,0,0), (1,0,0)],'Red')
		show_color_map(total[:,:,0], cm)
		plt.title('Red channel')

		plt.subplot(2, 2, 3)
		cm = color_map([(0,0,0), (0,1,0)],'Green')
		show_color_map(total[:,:,1], cm)
		plt.title('Green channel')

		plt.subplot(2, 2, 4)
		cm = color_map([(0,0,0), (0,0,1)],'Blue')
		show_color_map(total[:,:,2], cm)
		plt.title('Blue channel')

		fig = plt.figure()
		# global img_index
		fig.canvas.manager.set_window_title(f'{img_index}: Final')
		img_index += 1
		display_image(total)

	if verbose:
		print("Converting array to image")

	if save_path is not None:
		if verbose:
			print("Saving the image at: " + save_path)
		plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
		if verbose:
			print("Image saved successfully")

	return total, Y

def get_diffs(y_original, y_decoded, visualize=False, save_path=None):
	diff = np.abs(y_original - y_decoded)

	if visualize:
		fig = plt.figure()
		global img_index
		fig.canvas.manager.set_window_title(f'{img_index}: Diffs')
		display_image(diff, cmap='gray')

	if save_path is not None:
		plt.imsave(save_path + 'Diffs.bmp', diff, cmap='gray')

	return diff

def mean_squared_error(original, decoded):

	original = original.astype(np.float64)
	decoded = decoded.astype(np.float64)

	nl = np.shape(original)[0]
	nc = np.shape(original)[1]
	return np.sum(np.subtract(original, decoded) ** 2) / (nl * nc)

def root_mean_squared_error(original, decoded):
	return np.sqrt(mean_squared_error(original, decoded))

def signal_to_noise_ratio(original, decoded):
	# convert to float64
	original = original.astype(np.float64)
	decoded = decoded.astype(np.float64)

	m = np.shape(original)[0]
	n = np.shape(original)[1]
	P = np.sum(original ** 2) / (m * n)
	return 10 * np.log10(P / mean_squared_error(original, decoded))

def peak_signal_to_noise_ratio(original, decoded):
	original = original.astype(np.float64)
	decoded = decoded.astype(np.float64)

	return 10 * np.log10(255 ** 2 / mean_squared_error(original, decoded))

def max_diff(original, decoded):
	return np.max(np.abs(original - decoded))

def avg_diff(original, decoded):
	original = original.astype(np.float64)
	decoded = decoded.astype(np.float64)

	return np.mean(np.abs(original - decoded))

def entropy(matrix):
    unique_elements, counts_elements = np.unique(matrix, return_counts=True)
    probabilities = counts_elements / np.sum(counts_elements)
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy