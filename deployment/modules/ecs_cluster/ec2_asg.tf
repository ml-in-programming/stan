resource "aws_autoscaling_group" "default" {
  name = "${var.resource_prefix}-autoscaling-group"
  launch_configuration = "${aws_launch_configuration.default.id}"
  max_size = "${var.aws_asg_max_size}"
  min_size = "${var.aws_asg_min_size}"
  availability_zones = [
    "${var.aws_availability_zones_names}"]
  vpc_zone_identifier = [
    "${var.aws_subnet_private_id}"]
}

resource "aws_autoscaling_policy" "scale_up" {
  name = "${var.resource_prefix}-autoscaling-group-scale-up"
  scaling_adjustment = 1
  adjustment_type = "ChangeInCapacity"
  cooldown = 300
  autoscaling_group_name = "${aws_autoscaling_group.default.name}"
}

resource "aws_autoscaling_policy" "scale_down" {
  name = "${var.resource_prefix}-autoscaling-group-scale-down"
  scaling_adjustment = -1
  adjustment_type = "ChangeInCapacity"
  cooldown = 300
  autoscaling_group_name = "${aws_autoscaling_group.default.name}"
}
