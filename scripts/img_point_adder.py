import argparse
import matplotlib.pyplot as plt
import numpy as np
import requests

from PIL import Image


def main():
    clicked_points = []
    plotted_points = []
    def onclick(event):
        if event.inaxes == ax:
            # draw a red circle where the user clicks
            x, y = event.xdata, event.ydata
            clicked_points.append([int(x), int(y)])
            point, = ax.plot(x, y, 'ro')
            plotted_points.append(point)
            fig.canvas.draw_idle()
    
    def on_key_press(event):
        if event.key == "r":
            # remove all points
            clicked_points.clear()
            for point in plotted_points:
                point.remove()
            plotted_points.clear()
            fig.canvas.draw_idle()
            print("Points have been reset.")
        if event.key == "p":
            # print the current clicked points
            print("Clicked Points: ", clicked_points)

    parser = argparse.ArgumentParser(prog="Image Point Adder")
    parser.add_argument("--path", type=str, default=None, help="path of the image")
    parser.add_argument("--url", type=str, default=None, help="url of the image")

    args = parser.parse_args()

    img = None

    if args.path is None:
        img = Image.open(requests.get(args.url, stream=True).raw).convert("RGB")
    elif args.url is None:
        img = Image.open(args.path).convert("RGB")

    img = np.array(img)

    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.set_title("Click on the image to place points")

    clicked_points = []

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect("key_press_event", on_key_press)

    plt.show()
    print("Final Clicked Points: ", clicked_points)

if __name__ == "__main__":
    main()
