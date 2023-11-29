import os
import numpy as np
import imageio
import torch
import torchvision
import argparse
import cv2
from natsort import natsorted
from PIL import Image
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_image(path, image: np.ndarray) -> None:
  print(f'Saving {path}')
  if not path.parent.exists():
    path.parent.mkdir(exist_ok=True, parents=True)
  with path.open('wb') as f:
    image = Image.fromarray(np.asarray(image))
    image.save(f, format=path.suffix.lstrip('.'))


def image_to_uint8(image: np.ndarray) -> np.ndarray:
  """Convert the image to a uint8 array."""
  if image.dtype == np.uint8:
    return image
  if not issubclass(image.dtype.type, np.floating):
    raise ValueError(
        f'Input image should be a floating type but is of type {image.dtype!r}')
  return (image * 255).clip(0.0, 255).astype(np.uint8)


def make_divisible(image: np.ndarray, divisor: int) -> np.ndarray:
  """Trim the image if not divisible by the divisor."""
  height, width = image.shape[:2]
  if height % divisor == 0 and width % divisor == 0:
    return image

  new_height = height - height % divisor
  new_width = width - width % divisor

  return image[:new_height, :new_width]


def downsample_image(image: np.ndarray, scale: int) -> np.ndarray:
  """Downsamples the image by an integer factor to prevent artifacts."""
  if scale == 1:
    return image

  height, width = image.shape[:2]
  if height % scale > 0 or width % scale > 0:
    raise ValueError(f'Image shape ({height},{width}) must be divisible by the'
                     f' scale ({scale}).')
  out_height, out_width = height // scale, width // scale
  resized = cv2.resize(image, (out_width, out_height), cv2.INTER_AREA)
  return resized


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def load_data(args):
    path = args.path
    imgs = []
    try: # loading video path
        reader = imageio.get_reader(args.videopath)
        for i, im in enumerate(reader):
            imgs.append(im)
    except: # loading from folder of images
        if not args.mv_images:
          image_list = Path(path).glob('*.jpg')
          image_list = natsorted(image_list, key=str)
          print(image_list)
          for image_path in image_list:
              imgs.append(imageio.imread(image_path))
        else:
           times_list = Path(path).glob('*')
           times_list = natsorted(times_list, key=str)
           imgs_views = []
           for times_path in times_list:
              image_list = Path(times_path).glob('*.jpg')
              image_list = natsorted(image_list, key=str)
              print(image_list)
              for image_path in image_list:
                  imgs_views.append(imageio.imread(image_path))
              imgs.append(imgs_views)
    
    return np.array(imgs)

def multi_view_multi_time(args):
    """
    Generating multi view multi time data
    """

    Maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).cuda().eval()
    threshold = 0.5

    videoname, ext = os.path.splitext(os.path.basename(args.path))
    print(videoname)
    print(ext)
    if not args.mv_images:
      imgs = load_data(args)
      num_frames, H, W, _ = imgs.shape
      imgs = imgs[::int(np.ceil(num_frames / 100))]

      create_dir(os.path.join(args.data_dir, 'images'))
      create_dir(os.path.join(args.data_dir, 'images_colmap'))
      create_dir(os.path.join(args.data_dir, 'background_mask'))
      for idx, img in enumerate(imgs):
          print(idx)
          imageio.imwrite(os.path.join(args.data_dir, 'images', str(idx).zfill(5) + '.png'), img)
          imageio.imwrite(os.path.join(args.data_dir, 'images_colmap', str(idx).zfill(5) + '.jpg'), img)

          # Multiscale image
          image_scales = "1,2,4,8"  # @param {type: "string"}
          image_scales = [int(x) for x in image_scales.split(',')]

          tmp_rgb_dir = Path(os.path.join(args.data_dir, 'rgb'))
          print(tmp_rgb_dir)
          image = make_divisible(img, max(image_scales))
          for scale in image_scales:
              save_image(
                  tmp_rgb_dir / f'{scale}x/{str(idx).zfill(5)}.png',
                  image_to_uint8(downsample_image(image, scale)))

          # Get coarse background mask
          img = torchvision.transforms.functional.to_tensor(img).to(device)
          background_mask = torch.FloatTensor(H, W).fill_(1.0).to(device)
          objPredictions = Maskrcnn([img])[0]

          for intMask in range(len(objPredictions['masks'])):
              if objPredictions['scores'][intMask].item() > threshold:
                  if objPredictions['labels'][intMask].item() == 1: # person
                      background_mask[objPredictions['masks'][intMask, 0, :, :] > threshold] = 0.0

          background_mask_np = ((background_mask.cpu().numpy() > 0.1) * 255).astype(np.uint8)
          imageio.imwrite(os.path.join(args.data_dir, 'background_mask', str(idx).zfill(5) + '.jpg.png'), background_mask_np)
    else:
      times_list = Path(args.path).glob('*')
      times_list = natsorted(times_list, key=str)
      for time_idx, times_path in enumerate(times_list):
        image_list = Path(times_path).glob('*.jpg')
        image_list = natsorted(image_list, key=str)
        for view_idx, image_path in enumerate(image_list):
          img = imageio.imread(image_path)
          image_scales = "1"  # @param {type: "string"}
          image_scales = [int(x) for x in image_scales.split(',')]

          tmp_rgb_dir = Path(os.path.join(args.data_dir, 'mv_images'))
          print(tmp_rgb_dir)
          image = make_divisible(img, max(image_scales))
          for scale in image_scales:
              cam_idx = view_idx + 1
              filename = f'{scale}x/{str(time_idx).zfill(5)}/cam{cam_idx:02d}.png'
              save_image(
                  tmp_rgb_dir / filename,
                  image_to_uint8(downsample_image(image, scale)))
          


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str,
                        help='path, either video or folder of images')
    parser.add_argument("--data_dir", type=str, default='../data/',
                        help='where to store data')
    parser.add_argument("--mv_images", action="store_true")

    args = parser.parse_args()
    args.mv_images = True if 'mv_images' in args.path else False
    multi_view_multi_time(args)