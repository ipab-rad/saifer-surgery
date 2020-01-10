#include <ros/ros.h>
#include <geometry_msgs/WrenchStamped.h>
#include <sensor_msgs/JointState.h>
#include <trajectory_msgs/JointTrajectory.h>
#include <trajectory_msgs/JointTrajectoryPoint.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit/robot_state/robot_state.h>

#define FTHRESH 20.0

class IDyn {
	public:
		IDyn();
		~IDyn();
	private:
		std::string group_name;
		ros::NodeHandle n;
//		moveit::planning_interface::MoveGroupInterface *mg;			
		robot_model_loader::RobotModelLoader rl;
		robot_model::RobotModelPtr kinematic_model;
		robot_state::RobotStatePtr kinematic_state;

		trajectory_msgs::JointTrajectory base_traj;

		Eigen::MatrixXd Jinv;
		Eigen::MatrixXd Fbase;
		bool jac;
		bool ft_meas;

		ros::Subscriber sub1;
		ros::Subscriber sub2;
		ros::Subscriber sub_traj;
	
		void ft_callback(const geometry_msgs::WrenchStamped& msg);
		void jt_callback(const sensor_msgs::JointState& msg);
		void traj_callback(const trajectory_msgs::JointTrajectory& msg);

		std::vector<std::string> joint_names;
		std::vector<double> joint_angles;
		ros::Publisher pub;
	
		bool first;
		
};

IDyn::IDyn(void)
{
	//sub1 = n.subscribe("/red/robotiq_ft_wrench",1,&IDyn::ft_callback,this);		
	sub1 = n.subscribe("/red/robotiq_ft_wrench",1,&IDyn::ft_callback,this);	
	sub2 = n.subscribe("/joint_states",1,&IDyn::jt_callback,this);	
	sub_traj = n.subscribe("/red/trajectory",1,&IDyn::traj_callback,this);		
	
	pub = n.advertise<trajectory_msgs::JointTrajectory>("/red/vel_test",1);
	group_name = "red_arm";
//	first = true;
	Fbase = Eigen::MatrixXd(6,1);
	jac = false;
	ft_meas = false;

	rl = robot_model_loader::RobotModelLoader("robot_description");
	kinematic_model = rl.getModel();
	ros::spin();
}

IDyn::~IDyn(void)
{
//	delete mg;
}

void IDyn::traj_callback(const trajectory_msgs::JointTrajectory& msg)
{
	base_traj = msg;
	ROS_INFO("Got message");
}

void IDyn::ft_callback(const geometry_msgs::WrenchStamped& msg)
{
//	if (first)
//	{
//		Fbase << -msg.wrench.force.x, -msg.wrench.force.y, msg.wrench.force.z, msg.wrench.torque.x,msg.wrench.torque.y,msg.wrench.torque.z;
//		first = false;
//	}

	if (jac)
	{

		Eigen::MatrixXd ft(6,1);
		ft << -msg.wrench.force.x, -msg.wrench.force.y, msg.wrench.force.z, -msg.wrench.torque.x,-msg.wrench.torque.y,msg.wrench.torque.z;
		ROS_INFO("FT: %2.2f. Threshold: %2.2f.",ft.squaredNorm(),FTHRESH);
		//if ((Fbase-ft).squaredNorm() > 0)
		if (ft.squaredNorm() < FTHRESH)
		{
			ft = ft*0.0;
		}
		

		Eigen::MatrixXd K = Eigen::MatrixXd::Zero(6,6);
		for (int i = 0; i < 6; i++)
		{
			if (i < 3)
			{
				K(i,i) = 0.02;
			}
			else
			{
				K(i,i) = 0.0;
			}
		}
	
//		Eigen::MatrixXd twist = K*(Fbase-ft);
		Eigen::MatrixXd twist = -K*ft;
		Eigen::MatrixXd joint_vel = Jinv*twist; 
		
		if (base_traj.points.size() > 0)
		{	

			for (int j = 0; j < int(base_traj.points.size()); j++)
			{
				for (int i = 0; i < int(joint_vel.rows()); i++)
				{
					ROS_INFO("%d %d %f",j,i,joint_vel(i,0));
					base_traj.points[j].positions[i] = base_traj.points[j].positions[i] + 0.1*joint_vel(i,0);
					base_traj.points[j].velocities[i] = base_traj.points[j].velocities[i] + joint_vel(i,0);
					if (base_traj.points[j].velocities[i] > 0.5)
	        	                {
        			                base_traj.points[j].velocities[i] = 0.5;
                	        	}
	                        	if (base_traj.points[j].velocities[i] < -0.5)
		                        {
        		        	        base_traj.points[j].velocities[i] = -0.5;
                		        }
				}
				 
			}
			pub.publish(base_traj);
			base_traj.points.clear();
		}
		else
		{
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
		}

	}
}

void IDyn::jt_callback(const sensor_msgs::JointState& msg)
{
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
	ros::init(argc, argv, "inverse_dynamics");

	IDyn inverse_dynamics;

//	const robot_state::JointModelGroup* joint_model_group = move_group.getCurrentState()->getJointModelGroup(PLANNING_GROUP);
	return 0;
}
