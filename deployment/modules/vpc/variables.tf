variable "resource_prefix" {}

variable "az_count" {
  default = 2
}

variable "jetbrains_office_addrs_cidr_list" {
  default = "80.76.244.114/32,81.3.129.2/32,212.18.2.2/32,217.111.48.242/32,86.49.92.98/32,144.121.2.66/32"
  description = "Comma separated list of CIDRs of JetBrains offices"
}

variable "ingress_80_443_tcp_cidr_list" {
  description = "IPv4 Access from public Internet"
  default = "0.0.0.0/0"
}

variable "ingress_80_443_tcp_ipv6_cidr_list" {
  description = "IPv6 Access from public Internet"
  default = "::/0"
}
