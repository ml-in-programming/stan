output "ipv4_80_jb_security_group_id" {
  value = "${aws_security_group.ipv4_80_jb_security_group.id}"
}

output "all_ports_jb_security_group_id" {
  value = "${aws_security_group.all_ports_jb_security_group.id}"
}

output "ipv4_443_jb_security_group_id" {
  value = "${aws_security_group.ipv4_443_jb_security_group.id}"
}