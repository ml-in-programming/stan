resource "aws_security_group" "outer_security_group_80" {

  name = "${var.resource_prefix}-outer-security-group-80"
  vpc_id = "${var.vpc_id}"


  # HTTP access from the VPC
  ingress {
    from_port = 80
    to_port = 80
    protocol = "tcp"
    cidr_blocks = [
      "0.0.0.0/0"]
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

resource "aws_security_group" "outer_security_group_443" {

  name = "${var.resource_prefix}-outer-security-group-443"
  vpc_id = "${var.vpc_id}"


  # HTTP access from the VPC
  ingress {
    from_port = 443
    to_port = 443
    protocol = "tcp"
    cidr_blocks = [
      "0.0.0.0/0"]
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


