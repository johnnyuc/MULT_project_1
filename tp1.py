from tp1_functions import *

if __name__ == '__main__':

	qf_airport = 75
	save_path = "./imagens/airport_case/output/"
	visualize = True
	verbose = False
	Y_dpcm, Cb_dpcm, Cr_dpcm, height, width, y_original = encoder('imagens/airport_case/airport.bmp', qf=qf_airport, visualize=visualize, dct_blocks=8, cb_down_samp=2, cr_down_samp=2, interpolation=cv2.INTER_LINEAR, save_path=save_path, verbose=verbose)

	img, y_decoded = decoder(Y_dpcm, Cb_dpcm, Cr_dpcm, height, width, qf=qf_airport, visualize=visualize, verbose=verbose, dct_blocks=8, cb_down_samp=2, cr_down_samp=2, interpolation=cv2.INTER_LINEAR)
	diff = get_diffs(y_original, y_decoded, visualize=visualize, save_path=save_path)
	plt.show()


	original = read_im_2npyArray('imagens/airport_case/airport.bmp')
	mse = mean_squared_error(original, img)
	rmse = root_mean_squared_error(original, img)
	pnsr = peak_signal_to_noise_ratio(original, img)
	snr = signal_to_noise_ratio(original, img)
	max_d = max_diff(y_original, y_decoded)
	avg_df = avg_diff(y_original, y_decoded)
	print('MSE:', mse)
	print('RMSE', rmse)

	print('SNR:', snr)
	print('PSNR:', pnsr)
	print('Max Diff:', max_d)
	print('Average Diff:', avg_df)