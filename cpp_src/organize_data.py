import csv as csv


def organize_data():
    seen_large = False
    seen_small = False
    with open("out.txt", "r") as f:
        lines = f.readlines()
        with open("data.csv", "w", newline="") as csvfile:
            datawriter = csv.writer(
                csvfile, delimiter=" ", quotechar="|", quoting=csv.QUOTE_MINIMAL
            )
            for line in lines:
                line = line.replace("s", "")
                if "small" not in line and "large" not in line:
                    datawriter.writerow([line])


organize_data()
