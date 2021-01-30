"""
GUI to allow the user to manually change ROIs on a hand X-Ray image.
@author: Jordan Blackadar
"""
import pathlib
import tkinter
import tkinter.simpledialog
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageEnhance
from matplotlib.backend_bases import MouseButton
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from scan import Scan, Joint

prompts = ['mcp2', 'pip2', 'dip2',
           'mcp3', 'pip3', 'dip3',
           'mcp4', 'pip4', 'dip4',
           'mcp5', 'pip5', 'dip5']


class App(tkinter.Tk):
    """
    Class to extent tkinter.Tk for GUI
    """
    def __init__(self):
        # Properties and Attributes
        tkinter.Tk.__init__(self)
        self.attributes("-fullscreen", False)
        self.in_fullscreen = False
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
        self.im = self.plot.imshow(Image.open('icons/tufts.png'))
        self.points = []
        self.images = [tkinter.PhotoImage(file='icons/clear.png'),
                       tkinter.PhotoImage(file='icons/legend.png'),
                       tkinter.PhotoImage(file='icons/xray.png'),
                       tkinter.PhotoImage(file='icons/filesave.png'),
                       tkinter.PhotoImage(file='icons/zoom_crop.png'),
                       tkinter.PhotoImage(file='icons/zoom_joint.png'),
                       tkinter.PhotoImage(file='icons/zoom_home.png')]
        self.iconphoto(False, self.images[2])

        # Window and Canvas
        self.title("OA Hand QA")
        self.configure(bg='grey')
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=1)
        self.canvas.mpl_connect("scroll_event", self.on_scroll)
        self.canvas.mpl_connect("key_press_event", self.on_key)
        self.canvas.mpl_connect("button_press_event", self.on_click)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)

        # Bottom Toolbar
        toolbar = tkinter.Frame(self.master, bd=2, bg="grey", relief=tkinter.RAISED)
        clear_button = tkinter.Button(master=toolbar, image=self.images[0], command=self.clear_rois, bg="grey")
        legend_button = tkinter.Button(master=toolbar, image=self.images[1], command=self.toggle_legend, bg="grey")
        save_button = tkinter.Button(master=toolbar, image=self.images[3], command=self.save_scan, bg="grey")
        zoom_home = tkinter.Button(master=toolbar, image=self.images[6], command=self.zoom_home, bg="grey")
        zoom_one = tkinter.Button(master=toolbar, image=self.images[4], command=self.zoom_one, bg="grey")
        zoom_two = tkinter.Button(master=toolbar, image=self.images[5], command=self.zoom_two, bg="grey")
        self.prompt = tkinter.Label(master=toolbar, text="", bg="grey")
        self.xylabel = tkinter.Label(master=toolbar, text="", bg="grey")
        save_button.pack(side="left")
        clear_button.pack(side="left")
        legend_button.pack(side='left')
        zoom_home.pack(side="left")
        zoom_one.pack(side="left")
        zoom_two.pack(side="left")
        self.prompt.pack(side="left")
        self.xylabel.pack(side="right")
        toolbar.pack(side="bottom", fill=tkinter.X)

        # Menu Bar
        menubar = tkinter.Menu(self)
        filemenu = tkinter.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Open...", command=self.open_dir)
        filemenu.add_command(label="Start From...", command=self.start_from_prompt)
        filemenu.add_command(label="Save", command=self.save_scan)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.close)
        menubar.add_cascade(label="File", menu=filemenu)
        editmenu = tkinter.Menu(menubar, tearoff=0)
        editmenu.add_command(label="Clear ROIs", command=self.clear_rois)
        editmenu.add_command(label="Discard Changes", command=self.discard_changes)
        menubar.add_cascade(label="Edit", menu=editmenu)
        optionmenu = tkinter.Menu(menubar, tearoff=0)
        optionmenu.add_command(label="Fullscreen", command=self.toggle_fullscreen)
        optionmenu.add_command(label="Show/Hide Legend", command=self.toggle_legend)
        optionmenu.add_command(label="Autosave On/Off", command=self.toggle_autosave)
        menubar.add_cascade(label="Option", menu=optionmenu)
        helpmenu = tkinter.Menu(menubar, tearoff=0)
        helpmenu.add_command(label="Controls...", command=self.controls_box)
        helpmenu.add_command(label="About...", command=self.about_box)
        menubar.add_cascade(label="Help", menu=helpmenu)
        self.config(menu=menubar)

        self.toggle_autosave()  # Default to Autosave ON
        self.toggle_fullscreen()  # Default to Fullscreen ON

    @property
    def current_scan(self):
        """
        Property to access the currently selected scan based off of the index property.
        :return: Scan object
        """
        if len(self.scans) > 0:
            return self.scans[self.scans_index]
        else:
            return None

    @staticmethod
    def controls_box():
        """
        Accessed via Help -> Controls... to show the user how to use the program.
        """
        messagebox.showinfo(title="Controls", message=f"The following keyboard/mouse controls are available:\n\n"
                                                      f"Left Click: Select ROI to modify, or place ROI\n"
                                                      f"Right Click: Move Selected ROI to position\n"
                                                      f"WASD: Translate ROI along x/y\n"
                                                      f"<- , ->: Rotate ROI CCW/CW\n"
                                                      f"[ , ]: Decrease/Increase Contrast\n"
                                                      f"F12: Toggle Fullscreen View\n"
                                                      f"Q: Cycle through joint selection\n"
                                                      f"E: Next image in series\n"
                                                      f"X: Clear existing ROIs and enter ROI placement mode\n"
                                                      f"Z: Show/Hide Legend\n"
                                                      f"C: Save changes\n"
                                                      f"1: Zoom to Hand\n"
                                                      f"2: Zoom to Joints\n"
                                                      f"~\\`: Zoom Out")

    @staticmethod
    def about_box():
        """
        Accessed via Help -> About... to display info box.
        """
        messagebox.showinfo(title="About", message=f"Developed by Jordan Blackadar\n"
                                                   f"jordan.blackadar@outlook.com\n"
                                                   f"2021, Wentworth Institute")

    def toggle_fullscreen(self):
        """
        Toggles the fullscreen state of the program.
        Depending on the OS this will work better or worse.
        """
        if not self.in_fullscreen:
            self.attributes("-fullscreen", True)
            self.in_fullscreen = True
        else:
            self.attributes("-fullscreen", False)
            self.in_fullscreen = False

    def on_key(self, event):
        """
        Handles keypress events.
        :param event: Keypress Event from Matplotlib
        """
        # print(f"Key Press Event: {str(event.key)}")

        if event.key == "[":
            self.change_contrast(-0.1)
        elif event.key == "]":
            self.change_contrast(0.1)
        elif event.key == "w":
            self.translate_roi(0, -10)
        elif event.key == "s":
            self.translate_roi(0, 10)
        elif event.key == "a":
            self.translate_roi(-10, 0)
        elif event.key == "d":
            self.translate_roi(10, 0)
        elif event.key == "left":
            self.rotate_roi(-0.05)
        elif event.key == "right":
            self.rotate_roi(0.05)
        elif event.key == "f12":
            self.toggle_fullscreen()
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
        elif event.key == '1':
            self.zoom_one()
        elif event.key == '2':
            self.zoom_two()
        elif event.key == '`':
            self.zoom_home()

    def on_scroll(self, event):
        """
        Handles mousewheel (scroll) events.
        :param event: Matplotlib Event
        """
        # print(f"Scroll {event.button}")
        if self.input_override:
            self.bell()
            return
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
        """
        Handles mouse click events, including middle click.
        :param event: Matplotlib Event
        """
        # print(f"{event.button} Click: {event.xdata}, {event.ydata}")

        def find_nearest_center(x, y):
            """
            Finds the nearest Joint to (x, y) coordinates.
            :param x: int x coord
            :param y: int y coord
            :return: int index of nearest joint
            """
            xs = np.array([joint.x for joint in self.current_scan.joints])
            ys = np.array([joint.y for joint in self.current_scan.joints])
            distance_partial = (ys - y) ** 2 + (xs - x) ** 2
            idx = np.where(distance_partial == distance_partial.min())[0][0]
            return idx

        if event.button == MouseButton.MIDDLE:
            pass
        elif event.button == MouseButton.LEFT:
            if event.xdata is not None and event.ydata is not None:
                if not self.input_override and len(self.current_scan.joints) > 0:  # Selecting joint
                    nearest = find_nearest_center(event.xdata, event.ydata)
                    self.current_scan.selected_joint = nearest
                    for idx, joint in enumerate(self.current_scan.joints):
                        if idx == nearest:
                            joint.patch.set_edgecolor('yellow')
                        else:
                            joint.patch.set_edgecolor('red')
                    self.canvas.draw()
                else:
                    if not self.input_override:  # No joints existed but user clicked. Start collecting ROIs.
                        self.clear_rois()
                    collecting = prompts[self.input_override_index]
                    inp = Joint(int(event.xdata), int(event.ydata), 0.0, collecting)
                    self.current_scan.joints.append(inp)
                    self.current_scan.selected_joint = len(self.current_scan.joints) - 1
                    for idx, joint in enumerate(self.current_scan.joints):
                        if idx == self.current_scan.selected_joint:
                            joint.patch.set_edgecolor('yellow')
                        else:
                            joint.patch.set_edgecolor('red')
                    if self.input_override_index + 1 < len(prompts):
                        self.input_override_index += 1
                        self.display_prompt(f"Enter {prompts[self.input_override_index]} or edit")
                    else:
                        self.input_override_index = 0
                        self.input_override = False
                        self.display_prompt("")
                    self.redraw_scan()
        elif event.button == MouseButton.RIGHT:
            if self.current_scan.selected_joint is not None:
                if event.xdata is not None and event.ydata is not None:
                    self.move_roi(event.xdata, event.ydata)

    def on_motion(self, event):
        """
        Handles motion events, eg cursor moving over the window.
        This will happen all across the window but only has xydata if the cursor is over the plot.
        :param event: Matplotlib event
        """
        # print(f"Motion Event: {event.xdata}, {event.ydata}")
        x = event.xdata
        y = event.ydata
        if event.xdata is not None:
            self.xylabel.configure(text=f"({int(x)}, {int(y)})")
        else:
            self.xylabel.configure(text=f"")

    def open_dir(self):
        """
        Prompts the user for a directory, searches it for data, and preps the GUI to display it.
        Then calls redraw_scan() to start the image display.
        IMPORTANT: Assumes .txt infos are a subset of images. There cannot be a .txt without matching .png or this will
        ignore it.
        IMPORTANT: Assumes that sorting based on numerical ascension is implicitly provided by Pathlib via the OS.
        :return: None
        """
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
        messagebox.showinfo(title="Discovery", message=f"{len(intersection)} scan/label pairs will be loaded.\n"
                                                       f"{len(images)} images will be loaded.\n"
                                                       f"{len(infos)} info files were present.")

        sync_infos = []
        for image in images:
            # If the stem (only filename) of the image matches that of the info, append it, otherwise append None.
            # This is likely O(n^2), but is only performed once to load and therefore is tolerable compared to the
            # investment of more complicated logic.
            sync_infos.append(next((x for x in infos if x.stem == image.stem), None))

        for image, info in zip(images, sync_infos):
            self.scans.append(Scan.from_files(image, info))

        self.redraw_scan()

    def start_from_prompt(self):
        """
        Allows the user to enter a patient ID to find in the list of patients.
        Sets the user's current scan selection to that ID.
        :return: None
        """
        patient = tkinter.simpledialog.askstring("Custom Start", "Patient ID")
        patient = patient.strip()
        self.canvas.get_tk_widget().focus_force()

        found = False
        for idx, scan in enumerate(self.scans):
            if str(scan.patient).strip() == patient:
                self.scans_index = idx
                found = True
                break

        if not found:
            messagebox.showerror("Invalid Patient ID", f"No Patient with ID {patient} is loaded.")
            return
        else:
            self.redraw_scan()

    def redraw_scan(self):
        """
        Completely redraws the Matplotlib image plot and patches.
        This is unfortunately slow, but necessary, as the image size changes.
        """
        self.plot.clear()
        self.plot.imshow(self.current_scan.image, cmap='gist_gray')
        for joint in self.current_scan.joints:
            self.plot.add_patch(joint.patch)
            pt = self.plot.scatter(joint.x, joint.y, alpha=0.5, marker="o")
            pt.set_label(joint.label)
            self.points.append(pt)
        self.plot.axis(self.current_scan.axlimits)
        if self.legend:
            self.plot.legend()
        self.plot.set_title(f"Patient {self.current_scan.patient}" + (f" Visit {self.current_scan.visit}"
                                                                      if self.current_scan.visit != '' else ''))
        self.canvas.draw()

    def save_scan(self):
        """
        Saves the selected scan (if it was modified) and then marks the scan as unmodified.
        """
        if self.current_scan.modified:
            self.display_prompt(f"Saved {self.current_scan.patient}")
            print(f"Saving {self.current_scan}")
            self.current_scan.save()
            self.current_scan.modified = False

    def change_contrast(self, contrast_delta):
        """
        Scales the Scan by a small increment in contrast enhancement.
        This does not persist on the filesystem.
        :param contrast_delta: Float amount to change the contrast by
        """
        if self.current_scan.contrast_enhancement != 1.0:
            self.current_scan.image = self.current_scan.backup_image.copy()
        self.current_scan.backup_image = self.current_scan.image.copy()
        self.current_scan.contrast_enhancement += contrast_delta
        self.current_scan.image = ImageEnhance.Contrast(self.current_scan.image).enhance(
            self.current_scan.contrast_enhancement)
        self.display_prompt(f"Contrast {self.current_scan.contrast_enhancement: 0.2f}")
        self.redraw_scan()

    def clear_rois(self):
        """
        Clears all the ROIs of a Scan, allowing the user to input new ones.
        """
        self.current_scan.joints = []
        self.current_scan.selected_joint = None
        self.redraw_scan()
        self.input_override = True
        self.input_override_index = 0
        self.current_scan.modified = True
        self.display_prompt(f"Enter {prompts[self.input_override_index]}")

    def translate_roi(self, delta_x, delta_y):
        """
        Moves an ROI along the X or Y axis.
        :param delta_x: int, change in x
        :param delta_y: int, change in y
        """
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

    def rotate_roi(self, delta_theta):
        """
        Rotates the ROI on the xy plane.
        :param delta_theta: float Change in angle
        """
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

    def move_roi(self, new_x, new_y):
        """
        Moves an ROI to a custom new x,y coordinate.
        :param new_x: int, x position
        :param new_y: int, y position
        """
        sel = self.current_scan.selected_joint
        if sel is not None:
            joint = self.current_scan.joints[sel]
            print(f"Moving {joint} to ({int(new_x)}, {int(new_y)}).")
            joint.y = int(new_y)
            joint.x = int(new_x)
            joint.reload_patch()
            joint.patch.set_edgecolor('yellow')
            self.current_scan.modified = True
            self.redraw_scan()
        else:
            self.bell()

    def zoom_one(self):
        """
        Zooms the plot to only show the middle 70% of the y-axis.
        Meant to encompass only the hand.
        """
        im_size = self.current_scan.image.size
        # Remove top and bottom 15%
        y_size = im_size[1]
        y_clip = y_size * 0.15
        self.current_scan.axlimits = [0, self.current_scan.image.size[0], y_size - y_clip, y_clip]
        self.redraw_scan()

    def zoom_two(self):
        """
        Zooms the plot to only show the joints.
        Uses min/max x and y of the ROIs.
        """
        self.current_scan.set_axlimits_from_joints()
        self.redraw_scan()

    def zoom_home(self):
        """
        Resets zoom to normal.
        """
        self.current_scan.axlimits = [0, self.current_scan.image.size[0], self.current_scan.image.size[1], 0]
        self.redraw_scan()

    def toggle_legend(self):
        """
        Toggles displaying the legend on the plot.
        """
        if self.legend:
            self.plot.get_legend().remove()
            self.legend = False
        else:
            self.plot.legend()
            self.legend = True
        self.canvas.draw()

    def toggle_autosave(self):
        """
        Toggles autosave for scrolling and exit.
        """
        if self.autosave:
            self.autosave = False
            self.display_prompt("Autosave Disabled")
        else:
            self.autosave = True
            self.display_prompt("Autosave Enabled")

    def discard_changes(self):
        """
        Reloads Scan from file without saving changes.
        """
        self.scans[self.scans_index] = Scan.from_files(self.current_scan.image_path, self.current_scan.info_path)
        self.redraw_scan()

    def display_prompt(self, text):
        """
        Displays a string along the bottom toolbar.
        :param text: str, String to display on toolbar. Not guaranteed to persist.
        """
        self.prompt.configure(text=text)

    def close(self):
        """
        Saves if auto-save is on, checks for unsaved work, then quits.
        """
        if self.current_scan is not None:
            if self.autosave:
                self.save_scan()
        unsaved_work = [scan for scan in self.scans if scan.modified]
        if len(unsaved_work) > 0:
            result = messagebox.askquestion("Unsaved Changes", f"{len(unsaved_work)} scans were "
                                                               f"modified without saving. Save?",
                                            icon='warning')
            if result == 'yes':
                for scan in unsaved_work:
                    scan.save()
        self.quit()


if __name__ == "__main__":
    app = App()
    app.protocol('WM_DELETE_WINDOW', app.close)
    app.mainloop()
