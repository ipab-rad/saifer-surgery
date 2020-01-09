### Kinesthetic demonstration

<img align="right" alt="" src="https://github.com/ipab-rad/saifer-surgery/blob/master/src/saif_control/ft_compliance/ims/demo.gif" width="320" />Physically move red arm (only red has this functionality at present, to avoid risk of crushing blue camera). 

#### Make sure to zero the ft sensor before launching this node. External forces due to new tools, cables or payloads can cause unwanted motion. 

At present only x,y,z motion is allowed as orientation-based motion causes challenges due to mass differences from grasped tools/ unmodelled payloads.

Slight errors due to inverse dynamics jacobian linearisation can accumulate, so best practice is to only use this for short demonstrations, and re-zero the ft sensor regularly.


```
rosservice call /red/robotiq_ft_sensor_acc "command_id: 0 command: 'SET ZRO'"
roslaunch ft_compliance compliance.launch
```
