variable "resource_prefix" {}


variable "ec2_public_key_path" {}

variable "aws_ecs_ec2_instance_type" {
  description = "EC2 instance type that will be used for ECS"
}

variable "aws_asg_max_size" {
  description = "Required max size for ASG"
}

variable "aws_asg_min_size" {
  description = "Required min size for ASG"
}

variable "aws_availability_zones_names" {
  type = "list"
}

variable "aws_subnet_private_id" {
  type = "list"
}

variable "ec2_instance_security_group_ids" {
  type = "list"
}
