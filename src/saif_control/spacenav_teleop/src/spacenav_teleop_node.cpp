#include <ros/ros.h>
#include <sensor_msgs/JointState.h>
#include <sensor_msgs/Joy.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>
#include <Eigen/Eigen>
#include <Eigen/QR>
#include <robotiq_3f_gripper_articulated_msgs/Robotiq3FGripperRobotOutput.h>

class IDyn {
	public:
		IDyn();
		~IDyn();
	private:
		std::string group_name;
		ros::NodeHandle n;

		robot_model_loader::RobotModelLoader rl;
		robot_model::RobotModelPtr kinematic_model;
		robot_state::RobotStatePtr kinematic_state;

		Eigen::MatrixXd Jinv;

		ros::Subscriber sub1;
		ros::Subscriber sub2;
		void joy_callback(const sensor_msgs::Joy& msg);
		void jt_callback(const sensor_msgs::JointState& msg);

		std::vector<std::string> joint_names;
		std::vector<double> joint_angles;
		ros::Publisher pub;		
		ros::Publisher grip;

		bool jac;
};

IDyn::IDyn(void)
{
	sub1 = n.subscribe("/spacenav/joy",1,&IDyn::joy_callback,this);	
	sub2 = n.subscribe("/joint_states",1,&IDyn::jt_callback,this);	
	
	pub = n.advertise<trajectory_msgs::JointTrajectory>("/red/vel_test",1);
	grip = n.advertise<robotiq_3f_gripper_articulated_msgs::Robotiq3FGripperRobotOutput>("/Robotiq3FGripperRobotOutput",1);

	group_name = "red_arm";
	jac = false;

	rl = robot_model_loader::RobotModelLoader("robot_description");
	kinematic_model = rl.getModel();
	ros::spin();
}

IDyn::~IDyn(void)
{
//	delete mg;
}

void IDyn::joy_callback(const sensor_msgs::Joy& msg)
{
	if (jac)
	{
		if (msg.buttons[0])
		{
			robotiq_3f_gripper_articulated_msgs::Robotiq3FGripperRobotOutput cmd;
			cmd.rACT = 1;
			cmd.rPRA = 0;
			cmd.rGTO = 1;
			cmd.rSPA = 255;
			cmd.rFRA = 150;
			grip.publish(cmd);
			ros::Duration(0.1).sleep();
		}
		if (msg.buttons[1])
		{
			robotiq_3f_gripper_articulated_msgs::Robotiq3FGripperRobotOutput cmd;
			cmd.rACT = 1;
			cmd.rPRA = 255;
			cmd.rGTO = 1;
			cmd.rSPA = 255;
			cmd.rFRA = 150;	
			grip.publish(cmd);
			ros::Duration(0.1).sleep();
		}
		
		Eigen::MatrixXd ft(6,1);
		ft << msg.axes[0], msg.axes[1], msg.axes[2], msg.axes[3],msg.axes[4],msg.axes[5];

		if (ft.squaredNorm() > 0)
		{

			Eigen::MatrixXd K = Eigen::MatrixXd::Zero(6,6);
			for (int i = 0; i < 6; i++)
			{
				if (i < 3)
				{
					K(i,i) = 0.25;
				}
				else
				{
					K(i,i) = 0.5;
				}
			}
	
			Eigen::MatrixXd twist = K*(ft);
			Eigen::MatrixXd joint_vel = Jinv*twist; 

			trajectory_msgs::JointTrajectory jt;
			jt.header = msg.header;
			jt.joint_names =  joint_names;
			trajectory_msgs::JointTrajectoryPoint point;
			point.time_from_start = ros::Duration(1.0);
			point.positions = joint_angles;
	
			for (int i = 0; i < int(joint_vel.rows()); i++)
			{
				if (joint_vel(i,0) > 0.5)
				{
					joint_vel(i,0) = 0.5;
				}
				if (joint_vel(i,0) < -0.5)
				{
					joint_vel(i,0) = -0.5;
				}
				
				point.positions[i] = point.positions[i] + 0.1*joint_vel(i,0);
				point.velocities.push_back(joint_vel(i,0));
			}
			jt.points.push_back(point);	
			for (int i = 0; i < int(joint_vel.rows()); i++)
			{
				point.velocities[i] = 0.0;
			}	
			point.time_from_start = ros::Duration(2.0);	
			jt.points.push_back(point);	
			pub.publish(jt);
			ros::Duration(0.05).sleep();
 
			return;
		}
	}
}

void IDyn::jt_callback(const sensor_msgs::JointState& msg)
{
	if (!jac)
	{
		robotiq_3f_gripper_articulated_msgs::Robotiq3FGripperRobotOutput cmd;
		cmd.rACT = 1;
		cmd.rGTO = 1;
		cmd.rSPA = 255;
		cmd.rFRA = 150;
		grip.publish(cmd);
	
	}
//	ROS_INFO("Got joint message");
	robot_state::RobotStatePtr kinematic_state(new robot_state::RobotState(kinematic_model));
	kinematic_state->setToDefaultValues();
	const robot_state::JointModelGroup* joint_model_group = kinematic_model->getJointModelGroup(group_name);

	joint_names = joint_model_group->getVariableNames();
//	std::vector<double> joint_values;
//	joint_angles = msg.position;
	
	kinematic_state->setVariableValues(msg);
	kinematic_state->copyJointGroupPositions(joint_model_group, joint_angles);	
		
//	for (std::size_t i = 0; i < joint_names.size(); ++i)
//	{
//		ROS_INFO("Joint %s: %f", joint_names[i].c_str(), joint_values[i]);
//		ROS_INFO("Joint %s: %f", msg.name[i].c_str(), msg.position[i]);
//	}

	Eigen::Vector3d reference_point_position(0.0, 0.0, 0.0);
	Eigen::MatrixXd jacobian;

	kinematic_state->getJacobian(joint_model_group, kinematic_state->getLinkModel(joint_model_group->getLinkModelNames().back()), reference_point_position, jacobian);	
	
	Jinv = jacobian.completeOrthogonalDecomposition().pseudoInverse();
//	ROS_INFO_STREAM("Jacobian: \n" << jacobian << "\n" << "PseudoInverse" << "\n" << Jinv << "\n");
	jac = true;
	return;
}


int main(int argc, char **argv)
{
	ros::init(argc, argv, "spacenav_teleop");

	IDyn inverse_dynamics;

//	const robot_state::JointModelGroup* joint_model_group = move_group.getCurrentState()->getJointModelGroup(PLANNING_GROUP);
	return 0;
}
