# Robotic ultrasound scanning

<img align="right" alt="" src="https://lh6.googleusercontent.com/mYc6gJawLcenGWlv45mC8hXxdiDjeRATbenvbSPODawNClQElz4GgsW3FcnjKNxTarLgniBweggy81NYMcXaOGLroyEmAknH6mjTj-dtPvzqPyDBFvA=w1280" width="400" />

This app uses Bayesian optimisation to move an ultrasound scanner over an ultrasound imaging phantom in search of a tumour like object.
The app relies on a reward model learned from demonstration ultrasound image sequences using the [visual IRL](https://github.com/ipab-rad/saifer-surgery/tree/irl/src/saif_learning/visual_irl) package. 

NB: the manipulator moves through a hard-coded volume of positions. This may need to be adjusted depending on the phantom position. We assume the ultrasound scanner is pre-grasped.


See [companion site](https://sites.google.com/view/ultrasound-scanner) for more details.

Launch robot and ultrasound image streamer
```
roslaunch saifer_launch dual.launch
roslaunch ultrasound_epiphan us.launch
```

Run scanning app
```
rosrun ultrasound_imager pairwise_ultrasound_scanner.py
```

The package also has some baselines (maximum entropy, servoing), but these don't work very well.
