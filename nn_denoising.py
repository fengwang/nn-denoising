'''
    <nn-denoising -- denoise heavily noised STEM images>
    Copyright (C) 2019 Feng Wang feng.wang@empa.ch

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from keras.models import load_model
import numpy as np
from skimage import io
import tifffile
import time


def norm(img):
    return (img - np.amin(img)) / (np.amax(img) - np.amin(img) + 1.0e-10)


def make_log(log, flag):
    if not flag:
        return
    print('[[Pred log]]', log, f'[[{time.ctime()}]]')


# TODO:
def padding( img ):
    pass

# TODO:
def unpadding( img, boarders ):
    pass


def predict(image_path, prediction_path=None, log_flag=False):
    prediction_path = prediction_path or (image_path + '_denoised.tif')
    make_log(
        f'executing denoising for image(s) at {image_path}, and saving denoised result to {prediction_path}',
        log_flag)

    # load Low Pass Filter Model
    lpf = load_model('./lpf.model')
    make_log('LPF model loaded', log_flag)
    # load Denoiser Model
    generator = load_model('./denoiser.model')
    make_log('denoiser model loaded', log_flag)

    # load experimental data
    # TODO: case of other format of input
    image = np.asarray(io.imread(image_path), dtype='float32')
    make_log(f'experimental data loaded from {image_path}', log_flag)
    if len(image.shape) == 2:
        image = image.reshape((1, ) + image.shape)
    number, row, col = image.shape
    image = norm( image )
    make_log('experimental data normalized', log_flag)

    # symmetric padding of experimental data
    image = np.pad(image, ((0, 0), (128, 128), (128, 128)), 'symmetric')
    make_log('experimental data padded', log_flag)
    image = image.reshape(image.shape + (1, ))

    # low-pass filter process
    proceed_image = lpf.predict(image)
    make_log('experimental data proceed by LPF', log_flag)

    # correct contrast
    for idx in range(number):
        for jdx in range(4):
            proceed_image[idx, :, :, jdx] = norm(proceed_image[idx, :, :, jdx])
    make_log('LPF data contrast fixed', log_flag)

    # denoising process
    prediction = generator.predict(proceed_image, batch_size=2)
    make_log('LPF data denoised', log_flag)

    # unpadding
    result = prediction[:, 128:128 + row, 128:128 + col, :]
    make_log('denoised data unpadded', log_flag)

    prediction_result = norm(result)
    prediction_result = np.asarray(
        prediction_result * (256 * 256 - 1), dtype='uint16')
    tifffile.imsave(prediction_path, prediction_result)
    make_log(f'unpaded data saved to {prediction_path}', log_flag)


if __name__ == '__main__':
    predict('./experimental.tif', log_flag=True)
    #predict('/home/feng/raid_storage/experimental_data/experimental/new_s22.tif', prediction_path='./128_new_s22_denoised.tif', log_flag=True)
    #predict('/home/feng/raid_storage/experimental_data/experimental/s12.tif', prediction_path='./256_s12_denoised.tif', log_flag=True)
    #predict('/home/feng/raid_storage/experimental_data/experimental/s25.tif', prediction_path='./512_s25_denoised.tif', log_flag=True)
    #predict('/home/feng/raid_storage/experimental_data/experimental/s8.tif', prediction_path='./1024_s8_denoised.tif', log_flag=True)

