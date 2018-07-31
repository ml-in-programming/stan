output "outer_security_group_id_80" {
  value = "${aws_security_group.outer_security_group_80.id}"
}

output "outer_security_group_id_443" {
  value = "${aws_security_group.outer_security_group_443.id}"
}