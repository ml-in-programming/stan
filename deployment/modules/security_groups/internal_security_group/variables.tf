variable "resource_prefix" {}

variable "vpc_id" {}

variable "alb_security_group" {}

variable "vpc_ip_range" {
  default = "10.10.0.0/16"
}