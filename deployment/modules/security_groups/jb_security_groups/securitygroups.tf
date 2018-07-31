resource "aws_security_group" "ipv4_80_jb_security_group" {
  name = "${var.resource_prefix}-ipv4-80-jb-security-group"
  description = "ipv4-80-jb-security-group (managed by Terraform)"
  vpc_id = "${var.vpc_id}"

  ingress {
    from_port = 80
    to_port = 80
    protocol = "tcp"
    cidr_blocks = [
      "80.76.244.114/32",
      "81.3.129.2/32",
      "212.18.2.2/32",
      "217.111.48.242/32",
      "86.49.92.98/32",
      "144.121.2.66/32",
      "195.144.231.194/32",
      "217.148.215.18/32"]

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


resource "aws_security_group" "ipv4_443_jb_security_group" {
  name = "${var.resource_prefix}-ipv4-443-jb-security-group"
  description = "ipv4-443-jb-security-group (managed by Terraform)"
  vpc_id = "${var.vpc_id}"

  ingress {
    from_port = 443
    to_port = 443
    protocol = "tcp"
    cidr_blocks = [
      "80.76.244.114/32",
      "81.3.129.2/32",
      "212.18.2.2/32",
      "217.111.48.242/32",
      "86.49.92.98/32",
      "144.121.2.66/32",
      "195.144.231.194/32",
      "217.148.215.18/32"]

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

resource "aws_security_group" "all_ports_jb_security_group" {
  name = "${var.resource_prefix}-all-ports-jb-security-group"
  description = "all-ports-jb-security-group (managed by Terraform)"
  vpc_id = "${var.vpc_id}"

  ingress {
    from_port = 0
    to_port = 65535
    protocol = "tcp"
    cidr_blocks = [
      "80.76.244.114/32",
      "81.3.129.2/32",
      "212.18.2.2/32",
      "217.111.48.242/32",
      "86.49.92.98/32",
      "144.121.2.66/32",
      "195.144.231.194/32",
      "217.148.215.18/32"]

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



