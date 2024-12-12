import numpy as np
from vispy import app, gloo
from vispy.scene import SceneCanvas
# from vispy.util.transforms import perspective, translate, rotate
from vispy.scene.visuals import Markers
from vispy.visuals.transforms import STTransform

import sys

def launch_vispy_window(all_poss_with_time):
    print(all_poss_with_time.shape)
    all_poss_with_time = all_poss_with_time/10.0
    n_dims = all_poss_with_time.shape[2]



    assert n_dims == 2 or n_dims == 3, "Only 2D or 3D data is supported"

    if n_dims == 2:
        n_dims = 3
        all_poss_with_time = np.concatenate([all_poss_with_time, np.zeros((all_poss_with_time.shape[0], 1, 2))], axis=1)

    n_points = all_poss_with_time.shape[1]
    n_times = all_poss_with_time.shape[0]

    # Create a canvas with a 3D viewport
    canvas = SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()   
    # view.camera.fov = 45
    view.camera = 'fly'
    # km = view.camera._keymap 
    # km.pop('E')
    # km.pop('Q')
    # view.camera._keymap = km                           

    # add the scatter plot
    scatter = Markers(scaling=True, spherical=True)
    scatter.set_data(all_poss_with_time[0],edge_width=0, edge_color=None, face_color=(0, 0, 1, 1), size=.1)

    view.add(scatter)

    t = 0
    def step(event):
        nonlocal t
        t += 5
        t = t % n_times
        scatter.set_data(all_poss_with_time[t], edge_width=0, edge_color=None, face_color=(0, 0, 1, 1), size=.1)

    
    # Implement key presses
    @canvas.connect
    def on_key_press(event):
        if event.text == ' ':
            if timer.running:
                timer.stop()
            else:
                timer.start()

    timer = app.Timer(interval=1./10.)
    timer.connect(step)
    timer.start()


    canvas.show()
    app.run()


if __name__ == "__main__":
    # get command line arguments
    if len(sys.argv) != 2:
        print("Usage: python visualize.py <path_to_data>")
        sys.exit(1)

    path_to_data = sys.argv[1]

    data = np.load(path_to_data + ".npy")

    launch_vispy_window(data)