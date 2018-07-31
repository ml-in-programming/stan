output "alb_dns_name" {
  value = "${module.nginx-es-proxy-nlb.alb_dns_name}"
}