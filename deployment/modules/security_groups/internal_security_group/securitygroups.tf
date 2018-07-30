resource "aws_security_group" "internal_security_group" {

  name = "${var.resource_prefix}-private-security-group"
  vpc_id = "${var.vpc_id}"


  # HTTP access from the VPC
  ingress {
    from_port = 1
    to_port = 65535
    protocol = "tcp"
    cidr_blocks = [
      "${var.vpc_ip_range}"]
    security_groups = [
      "${var.alb_security_group}"]
  }

  # outbound internet access
  egress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    cidr_blocks = [
      "0.0.0.0/0"]
  }
}


