from PIL import Image
import os


def scale_down(img_path, save_path, size, speed=100, quality=85):
    '''
    :func: 按比例缩小图片
    :param img_path: 原图路径
    :param save_path: 保存路径
    :param size: 目的大小，以kb为单位
    :param speed: 缩小速度，过小时会引起震荡
    :param quality: 保存图像的质量，值的范围从1（最差）到95（最佳）;
        使用中应尽量避免高于95的值; 100会禁用部分JPEG压缩算法，并导致大文件图像质量几乎没有任何增益。
        在图像过分缩小的情况下可以将设置成比较大的quality值；
    '''
    img = Image.open(img_path)
    img.save(save_path, 'JPEG', quality=quality)

    while os.path.getsize(save_path) > size * 1024:
        # (width, heighth)
        width, height = img.size[0] - speed, img.size[1] - speed
        img.thumbnail((width, height), Image.ANTIALIAS)
        img.save(save_path, 'JPEG', quality=quality)
        # print('width * height:', width, height, '\tszie:', os.path.getsize(save_path))

    print('It has finished.\nthe save_path: {0}\nthe final size: {1}kb\nwidth * height: {2} * {3}'
          .format(save_path, os.path.getsize(save_path) / 1024, img.size[0], img.size[1]))


if __name__ == '__main__':
    scale_down('pics/DSC05852.jpg', './cns_.jpg', 250)