from PIL import Image


def reconstruct_image(filename, dest_filename):
    width, height = (2000, 1000)

    new_image = Image.new("RGB", (width, height))
    data = new_image.load()

    with open(filename, "r") as img:
        counter = 0
        lines = img.readlines()
        for line in lines:
            if "strong" in line:
                data[(counter // height, counter % height)] = (245, 66, 66)
            elif "mid" in line:
                data[(counter // height, counter % height)] = (81, 66, 245)
            elif "weak" in line:
                data[(counter // height, counter % height)] = (0, 0, 0)
            counter += 1

    new_image.save(dest_filename, "png")


reconstruct_image("img.txt", "img.png")
reconstruct_image("thresholded.txt", "thresholded.png")
