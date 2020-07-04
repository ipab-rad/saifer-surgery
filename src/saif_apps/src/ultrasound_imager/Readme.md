# Robotic ultrasound scanning

<img align="right" alt="" src="https://github.com/ipab-rad/saifer-surgery/blob/master/docs/images/scan.gif" width="150"/>

This app uses Bayesian optimisation to move an ultrasound scanner over an ultrasound imaging phantom in search of a tumour like object.
The app relies on a reward model learned from demonstration ultrasound image sequences using the [visual IRL](https://github.com/ipab-rad/visual_irl/tree/4bd514caab754971353f7e77a481f564f747c311) package. 

NB: the manipulator moves through a hard-coded volume of positions. This may need to be adjusted depending on the phantom position. We assume the ultrasound scanner is pre-grasped.


See [companion site](https://sites.google.com/view/ultrasound-scanner) for more details.

Launch robot and ultrasound image streamer
```
roslaunch saifer_launch dual_arm.launch
roslaunch ultrasound_epiphan us.launch
```

Run scanning app
```
rosrun ultrasound_imager pairwise_ultrasound_scanner.py
```

The package also has some baselines (maximum entropy, servoing), but these don't work very well.
