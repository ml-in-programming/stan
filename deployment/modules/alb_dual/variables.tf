variable "alb_name" {}

variable "vpc_id" {}

variable "aws_subnet_public_id" {
  type = "list"
}

variable "alb_security_groups" {
  type = "list"
}


variable "target_group_arn" {}

variable "certificate_name" {}
