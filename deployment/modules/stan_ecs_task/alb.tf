module "nginx-es-proxy-nlb" {
  source = "../alb_dual"
  alb_name = "${var.resource_prefix}-${var.internal_prefix}-proxy-alb"

  aws_subnet_public_id = "${var.public_subnet_id}"
  alb_security_groups = "${var.alb_security_group}"

  vpc_id = "${var.vpc_id}"
  target_group_arn = "${aws_alb_target_group.default.arn}"
}