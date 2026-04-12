from PIL import Image


def deconstruct_image(filename, dest_filename):
    im = Image.open(filename)    
    pixels = list(im.getdata())
    width, height = im.size
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
    print("Width of image is: " + str(width) + ", Height is: " + str(height))
    with open(dest_filename, "w") as f:
        for pixel in pixels:
            for pixel_sub in pixel:
                pixel = int(pixel_sub[0] << 16) | int(pixel_sub[1] << 8) | int(pixel_sub[2])
                f.write(str(pixel))
                f.write("\n")

deconstruct_image("imgs/0001.png", "out/0001.txt")