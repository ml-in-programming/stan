variable "resource_prefix" {}
variable "internal_prefix" {
  default = "elk"
}

variable "zone_id" {}
variable "dns_name" {}

variable "alb_security_group" {
  type = "list"
}

variable "ecs_cluster_id" {}
variable "vpc_id" {}
variable "public_subnet_id" {
  type = "list"
}

variable "aws_region" {}