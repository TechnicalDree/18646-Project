from PIL import Image

def get_rgb(pixel_val):
     return ((pixel_val & 0xFF0000) >> 16, (pixel_val & 0xFF00) >> 8, (pixel_val) & 0xFF)

def reconstruct_image(img_filename, txt_filename, dest_filename):
    img = Image.open(img_filename)
    width, height = img.size
    print("Width of image is: " + str(width) + ", Height is: " + str(height))

    new_image = Image.new("RGB", (width, height))
    data = new_image.load()
    with open(txt_filename, "r") as img:
        counter = 0
        lines = (img.readlines()[0]).split("|")
        for line in lines:
                data[(counter % width, counter // width)] = get_rgb(int(line))
                counter += 1

    new_image.save(dest_filename, "png")

reconstruct_image("imgs/0001.png", "out/modified_img.txt", "out/0001_modified.png")