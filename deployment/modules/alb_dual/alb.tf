resource "aws_lb" "alb" {
  name = "${var.alb_name}"
  internal = false
  ip_address_type = "dualstack"

  enable_deletion_protection = true

  subnets = [
    "${var.aws_subnet_public_id}"]
  security_groups = [
    "${var.alb_security_groups}"]
  idle_timeout = "240"

  tags {
    Name = "${var.alb_name}"
  }
}

//Listener for 443 port of alb targeted at active target group arn of which is written in consul
resource "aws_lb_listener" "listener_443" {
  load_balancer_arn = "${aws_lb.alb.arn}"
  port = "443"
  ssl_policy = "ELBSecurityPolicy-2016-08"
  certificate_arn = "${data.aws_acm_certificate.certificate.arn}"
  protocol = "HTTPS"

  default_action {
    target_group_arn = "${var.target_group_arn}"
    type = "forward"
  }
}

//Listener for 80 port of alb targeted at active target group arn of which is written in consul
resource "aws_lb_listener" "listener_80" {
  load_balancer_arn = "${aws_lb.alb.arn}"
  port = "80"
  protocol = "HTTP"

  #should be changed to redirect by hands now, cause terraform not supports redirect for alb for now
  default_action {
    target_group_arn = "${var.target_group_arn}"
    type = "forward"
  }
}