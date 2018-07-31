data "aws_route53_zone" "ml_zone" {
  name = "ml.aws.intellij.net."
}

module "vpc_stan" {
  source = "modules/vpc"
  az_count = 2
  resource_prefix = "stan"
}

module "ecs_cluster" {
  source = "modules/ecs_cluster"
  aws_ecs_ec2_instance_type = "t2.micro"
  aws_subnet_private_id = "${module.vpc_stan.aws_subnet_private_id}"
  resource_prefix = "stan"
  ec2_instance_security_group_ids = ["${module.internal_security_group.internal_security_group_id}"]
  ec2_public_key_path = "stan_ec2.pub"
  aws_availability_zones_names = "${module.vpc_stan.aws_availability_zones_names}"
  aws_asg_min_size = "1"
  aws_asg_max_size = "1"
}

module "stan_task" {
  source = "modules/stan_ecs_task"
  ecs_cluster_id = "${module.ecs_cluster.ecs_cluster_id}"
  alb_security_group = ["${module.outer_security_groups.outer_security_group_id_80}", "${module.outer_security_groups.outer_security_group_id_443}"]
  resource_prefix = "stan"
  vpc_id = "${module.vpc_stan.vpc_id}"
  public_subnet_id = "${module.vpc_stan.aws_subnet_public_id}"
  aws_region = "${local.region}"
  dns_name = "stan.ml.aws.intellij.net"
  zone_id = "${data.aws_route53_zone.ml_zone.zone_id}"
}

module "jb_security_groups" {
  source = "modules/security_groups/jb_security_groups"

  resource_prefix = "stan-shared"

  vpc_id = "${module.vpc_stan.vpc_id}"
}

module "internal_security_group" {
  source = "modules/security_groups/internal_security_group"
  resource_prefix = "stan-shared"

  alb_security_group = "${module.jb_security_groups.ipv4_80_jb_security_group_id}"
  vpc_id = "${module.vpc_stan.vpc_id}"
}

module "outer_security_groups" {
  source = "modules/security_groups/outer_security_group"
  resource_prefix = "stan-shared"

  vpc_id = "${module.vpc_stan.vpc_id}"
}

