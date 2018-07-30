resource "aws_alb_target_group" "default" {
  name = "${var.resource_prefix}-${var.internal_prefix}-target-group"
  port = "80"
  protocol = "HTTP"
  vpc_id = "${var.vpc_id}"
  health_check {
    path = "/"
    protocol = "HTTP"
    healthy_threshold = 3
    unhealthy_threshold = 5
    interval = 10
    timeout = 5
    matcher = "200"
  }
  deregistration_delay = 30
}

