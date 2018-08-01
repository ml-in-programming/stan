resource "aws_route53_record" "alias_record" {
  name = "stan"
  type = "CNAME"
  zone_id = "${var.zone_id}"
  ttl = 60
  records = ["${module.service_alb.alb_dns_name}"]
}

module "service_alb" {
  source = "../alb_dual"
  alb_name = "${var.resource_prefix}-${var.internal_prefix}-alb"

  aws_subnet_public_id = "${var.public_subnet_id}"
  alb_security_groups = "${var.alb_security_group}"

  vpc_id = "${var.vpc_id}"
  target_group_arn = "${aws_alb_target_group.default.arn}"

  certificate_name = "${var.dns_name}"
}