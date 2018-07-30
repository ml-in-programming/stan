resource "aws_key_pair" "auth" {
  key_name = "${var.resource_prefix}-ec2-instance-key"
  public_key = "${file(var.ec2_public_key_path)}"
}

resource "aws_launch_configuration" "default" {
  name_prefix = "${var.resource_prefix}-launch-configuration"
  image_id = "${data.aws_ami.ecs.id}"
  instance_type = "${var.aws_ecs_ec2_instance_type}"
  security_groups = [
    "${var.ec2_instance_security_group_ids}"]
  iam_instance_profile = "${aws_iam_instance_profile.instance_profile.arn}"
  key_name = "${aws_key_pair.auth.key_name}"

  user_data = <<USERDATA
#!/bin/bash
echo ECS_CLUSTER=${aws_ecs_cluster.default.name} >> /etc/ecs/ecs.config
USERDATA
  root_block_device {
    volume_size = "20"
    volume_type = "gp2"
  }
  ebs_block_device {
    device_name = "/dev/xvdcz"
    // Disk naming is important!
    volume_size = "50"
    volume_type = "gp2"
  }

  lifecycle {
    create_before_destroy = true
  }
}

data "aws_ami" "ecs" {
  most_recent = true

  filter {
    name = "name"
    values = [
      "amzn-ami-*-ecs-optimized"]
  }

  filter {
    name = "architecture"
    values = [
      "x86_64"]
  }

  filter {
    name = "virtualization-type"
    values = [
      "hvm"]
  }

  filter {
    name = "owner-alias"
    values = [
      "amazon"]
  }

  owners = [
    "amazon"]
}
