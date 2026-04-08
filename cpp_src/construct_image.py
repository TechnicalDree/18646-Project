from PIL import Image


def reconstruct_image():
    width, height = (2000, 1000)

    new_image = Image.new("RGB", (width, height))
    data = new_image.load()

    with open("img.txt", "r") as img:
        counter = 0
        lines = img.readlines()
        for line in lines:
            if "strong" in line:
                data[(counter // height, counter % height)] = (255, 255, 255)
            elif "mid" in line:
                data[(counter // height, counter % height)] = (128, 128, 128)
            elif "weak" in line:
                data[(counter // height, counter % height)] = (0, 0, 0)
            counter += 1

    new_image.save("foo.png", "png")


reconstruct_image()
