import pathlib
import tkinter
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageEnhance
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure

from scan import Scan, Joint

prompts = ['mcp2', 'pip2', 'dip2',
           'mcp3', 'pip3', 'dip3',
           'mcp4', 'pip4', 'dip4',
           'mcp5', 'pip5', 'dip5']


class NavigationToolbar(NavigationToolbar2Tk):
    toolitems = [t for t in NavigationToolbar2Tk.toolitems if
                 t[0] not in ('Subplots',)]
    toolitems.extend([
            (None, None, None, None),
    ])

    def _init_toolbar(self):
        pass


class App(tkinter.Tk):

    def __init__(self):
        tkinter.Tk.__init__(self)
        self.attributes("-fullscreen", False)
        self.fullscreen_toggle = False
        self.input_override = False
        self.autosave = False
        self.input_override_index = 0
        self.directory = None
        self.scans = []
        self.scans_index = 0
        self.legend = True
        self.figure = Figure(dpi=100)
        self.figure.set_facecolor('grey')
        self.plot = self.figure.add_subplot(111)
        self.im = self.plot.imshow(Image.open('icons/tufts.png'))  # None if reusing imshow...
        self.points = []

        # Window Layout
        self.title("OA Hand QA")
        self.configure(bg='grey')
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)

        # Interactive Elements
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("key_press_event", self.on_key)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        toolbar = NavigationToolbar(self.canvas, self)
        toolbar.children['!button4'].config(command=self.save_scan)
        self.images = [tkinter.PhotoImage(file='icons/clear.png'),
                       tkinter.PhotoImage(file='icons/legend.png'),
                       tkinter.PhotoImage(file='icons/xray.png')]
        self.iconphoto(False, self.images[2])
        clear_button = tkinter.Button(master=toolbar, image=self.images[0], command=self.clear_rois)
        legend_button = tkinter.Button(master=toolbar, image=self.images[1], command=self.toggle_legend)
        self.prompt = tkinter.Label(master=toolbar, text="")
        clear_button.pack(side="left")
        legend_button.pack(side='left')
        self.prompt.pack(side="left")
        toolbar.update()

        # Menu Bar
        menubar = tkinter.Menu(self)
        filemenu = tkinter.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open...", command=self.open_dir)
        filemenu.add_command(label="Save", command=self.save_scan)
        filemenu.add_command(label="Fullscreen", command=self.fullscreen)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        editmenu = tkinter.Menu(menubar, tearoff=0)
        editmenu.add_command(label="Clear All", command=self.clear_rois)
        editmenu.add_command(label="Toggle Legend", command=self.toggle_legend)
        editmenu.add_command(label="Toggle Autosave", command=self.toggle_autosave)
        menubar.add_cascade(label="Edit", menu=editmenu)
        helpmenu = tkinter.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Controls...", command=self.controls_box)
        helpmenu.add_command(label="About...", command=self.about_box)
        menubar.add_cascade(label="Help", menu=helpmenu)
        self.config(menu=menubar)

    @property
    def current_scan(self):
        if len(self.scans) > 0:
            return self.scans[self.scans_index]
        else:
            return None

    @staticmethod
    def controls_box():
        messagebox.showinfo(title="Controls", message=f"The following keyboard/mouse controls are available:\n\n"
                                                      f"Middle Click: Select ROI to modify, or place ROI\n"
                                                      f"WASD: Translate ROI along x/y\n"
                                                      f"<- , ->: Rotate ROI CCW/CW\n"
                                                      f"[ , ]: Decrease/Increase Contrast\n"
                                                      f"F12: Toggle Fullscreen View\n"
                                                      f"Q: Cycle through joint selection\n"
                                                      f"E: Next image in series\n"
                                                      f"X: Clear existing ROIs and enter ROI placement mode\n"
                                                      f"Z: Show/Hide Legend\n"
                                                      f"C: Save changes")

    @staticmethod
    def about_box():
        messagebox.showinfo(title="About", message=f"Developed by Jordan Blackadar\n"
                                                   f"jordan.blackadar@outlook.com\n"
                                                   f"2021, Wentworth Institute")

    def fullscreen(self):
        if not self.fullscreen_toggle:
            self.attributes("-fullscreen", True)
            self.fullscreen_toggle = True
        else:
            self.attributes("-fullscreen", False)
            self.fullscreen_toggle = False

    def on_key(self, event):
        # print(f"Key Press Event: {str(event.key)}")
        def translate_roi(delta_x, delta_y):
            sel = self.current_scan.selected_joint
            if sel is not None:
                joint = self.current_scan.joints[sel]
                joint.y += delta_y
                joint.x += delta_x
                joint.reload_patch()
                joint.patch.set_edgecolor('yellow')
                self.current_scan.modified = True
                self.redraw_scan()
            else:
                self.bell()

        def rotate_roi(delta_theta):
            sel = self.current_scan.selected_joint
            if sel is not None:
                joint = self.current_scan.joints[sel]
                joint.angle += delta_theta
                joint.reload_patch()
                joint.patch.set_edgecolor('yellow')
                self.current_scan.modified = True
                self.redraw_scan()
            else:
                self.bell()

        if event.key == "[":
            self.change_contrast(-0.1)
        elif event.key == "]":
            self.change_contrast(0.1)
        elif event.key == "w":
            translate_roi(0, -10)
        elif event.key == "s":
            translate_roi(0, 10)
        elif event.key == "a":
            translate_roi(-10, 0)
        elif event.key == "d":
            translate_roi(10, 0)
        elif event.key == "left":
            rotate_roi(-0.1)
        elif event.key == "right":
            rotate_roi(0.1)
        elif event.key == "f12":
            self.fullscreen()
        elif event.key == "q":
            if self.current_scan.selected_joint is not None:
                if self.current_scan.selected_joint < len(self.current_scan.joints) - 1:
                    self.current_scan.selected_joint += 1
                else:
                    self.current_scan.selected_joint = 0
            else:
                self.current_scan.selected_joint = 0
            for idx, joint in enumerate(self.current_scan.joints):
                if idx == self.current_scan.selected_joint:
                    joint.patch.set_edgecolor('yellow')
                else:
                    joint.patch.set_edgecolor('red')
            self.canvas.draw()
        elif event.key == "e":
            if self.scans_index < len(self.scans) - 1:
                if self.autosave:
                    self.save_scan()
                self.scans_index += 1
                self.redraw_scan()
            else:
                self.bell()
        elif event.key == 'x':
            self.clear_rois()
        elif event.key == 'z':
            self.toggle_legend()
        elif event.key == 'c':
            self.save_scan()

    def on_scroll(self, event):
        # print(f"Scroll {event.button}")

        if event.button == "up":
            if self.scans_index > 0:
                if self.autosave:
                    self.save_scan()
                self.scans_index -= 1
                self.redraw_scan()
            else:
                self.bell()
        elif event.button == "down":
            if self.scans_index < len(self.scans) - 1:
                if self.autosave:
                    self.save_scan()
                self.scans_index += 1
                self.redraw_scan()
            else:
                self.bell()

    def on_click(self, event):
        # print(f"{event.button} Click: {event.xdata}, {event.ydata}")

        def find_nearest_center(x, y):
            xs = np.array([joint.x for joint in self.current_scan.joints])
            ys = np.array([joint.y for joint in self.current_scan.joints])
            distance_partial = (ys - y) ** 2 + (xs - x) ** 2
            idx = np.where(distance_partial == distance_partial.min())[0][0]
            return idx

        if event.button == MouseButton.LEFT:
            pass
        elif event.button == MouseButton.MIDDLE:
            if not self.input_override:
                nearest = find_nearest_center(event.xdata, event.ydata)
                self.current_scan.selected_joint = nearest
                for idx, joint in enumerate(self.current_scan.joints):
                    if idx == nearest:
                        joint.patch.set_edgecolor('yellow')
                    else:
                        joint.patch.set_edgecolor('red')
                self.canvas.draw()
            else:
                collecting = prompts[self.input_override_index]
                inp = Joint(event.xdata, event.ydata, 0.0, collecting)
                self.current_scan.joints.append(inp)
                self.current_scan.selected_joint = len(self.current_scan.joints) - 1
                if self.input_override_index + 1 < len(prompts):
                    self.input_override_index += 1
                    self.display_prompt(f"Enter {prompts[self.input_override_index]} or Edit")
                else:
                    self.input_override_index = 0
                    self.input_override = False
                    self.display_prompt("")
                self.redraw_scan()
        elif event.button == MouseButton.RIGHT:
            pass

    def open_dir(self):
        self.plot.clear()
        self.scans = []
        self.scans_index = 0
        self.directory = pathlib.Path(filedialog.askdirectory())
        images = list(self.directory.glob(f"*.png"))
        infos = list(self.directory.glob(f"*.txt"))

        if len(images) == 0:
            messagebox.showerror(title="Error Opening Directory", message=f"{self.directory} has no images.")
            return
        elif len(infos) == 0:
            messagebox.showerror(title="Error Opening Directory", message=f"{self.directory} has no info files.")
            return

        intersection = set([path.stem for path in images]) & set(path.stem for path in infos)
        messagebox.showinfo(title="Discovery", message=f"{len(intersection)} scans will be loaded.\n"
                                                       f"{len(images)} images were present.\n"
                                                       f"{len(infos)} info files were present.")
        images = [image for image in images if image.stem in intersection]
        infos = [info for info in infos if info.stem in intersection]

        for image, info in zip(images, infos):
            self.scans.append(Scan.from_files(image, info))

        self.redraw_scan()

    def redraw_scan(self):
        self.plot.clear()
        self.plot.imshow(self.current_scan.image, cmap='gist_gray')
        for joint in self.current_scan.joints:
            self.plot.add_patch(joint.patch)
            pt = self.plot.scatter(joint.x, joint.y, alpha=0.5, marker="o")
            pt.set_label(joint.label)
            self.points.append(pt)
        if self.legend:
            self.plot.legend()
        self.plot.set_title(f"Patient {self.current_scan.patient}")
        self.canvas.draw()

    def redraw_scan_minimal(self):
        [p.remove() for p in reversed(self.plot.patches)]  # Remove all plotted patches
        [pt.remove() for pt in self.points]  # Remove plotted center points
        self.points = []  # Set the window's list of center points to empty
        if self.im is None:  # If there was no existing image plotted, plot one
            self.im = self.plot.imshow(self.current_scan.image, cmap='gist_gray')
        else:  # Only replace the image data and update the extent
            image = self.current_scan.image
            width, height = image.size
            self.im.set_data(image)
            self.im.set_extent((width, 0, height, 0))
            self.im.set_cmap('gist_gray')
            # self.im.autoscale()
        for joint in self.current_scan.joints:
            self.plot.add_patch(joint.patch)
            pt = self.plot.scatter(joint.x, joint.y, alpha=0.5, marker="o")
            pt.set_label(joint.label)
            self.points.append(pt)
        if self.legend:
            self.plot.legend()
        self.plot.set_title(f"Patient {self.current_scan.patient}")
        self.canvas.draw()

    def redraw_annotations(self):
        [p.remove() for p in reversed(self.plot.patches)]
        [pt.remove() for pt in self.points]
        self.points = []
        for joint in self.current_scan.joints:
            self.plot.add_patch(joint.patch)
            pt = self.plot.scatter(joint.x, joint.y, alpha=0.5, marker="o")
            pt.set_label(joint.label)
            self.points.append(pt)
        if self.legend:
            self.plot.legend()
        self.canvas.draw()

    def save_scan(self):
        if self.current_scan.modified:
            self.display_prompt(f"Saved {self.current_scan.patient}")
            print(f"Saving {self.current_scan}")
            self.current_scan.save()
            self.current_scan.modified = False

    def change_contrast(self, contrast_delta):
        if self.current_scan.contrast_enhancement != 1.0:
            self.current_scan.image = self.current_scan.backup_image.copy()
        self.current_scan.backup_image = self.current_scan.image.copy()
        self.current_scan.contrast_enhancement += contrast_delta
        self.current_scan.image = ImageEnhance.Contrast(self.current_scan.image).enhance(
            self.current_scan.contrast_enhancement)
        self.redraw_scan()

    def clear_rois(self):
        self.current_scan.joints = []
        self.redraw_scan()
        self.input_override = True
        self.display_prompt(f"Enter {prompts[self.input_override_index]}")

    def toggle_legend(self):
        if self.legend:
            self.plot.get_legend().remove()
            self.legend = False
        else:
            self.plot.legend()
            self.legend = True
        self.canvas.draw()

    def toggle_autosave(self):
        if self.autosave:
            self.autosave = False
            self.display_prompt("Autosave Disabled")
        else:
            self.autosave = True
            self.display_prompt("Autosave Enabled")

    def display_prompt(self, text):
        self.prompt.configure(text=text)


if __name__ == "__main__":
    app = App()
    app.mainloop()
