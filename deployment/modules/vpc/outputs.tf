output "vpc_id" {
  value = "${aws_vpc.default.id}"
}

output "aws_subnet_public_id" {
  value = "${aws_subnet.public.*.id}"
}

output "aws_subnet_private_id" {
  value = "${aws_subnet.private.*.id}"
}

output "aws_availability_zones_names" {
  value = [
    "${data.aws_availability_zones.available.names[0]}",
    "${data.aws_availability_zones.available.names[1]}"
  ]
}

output "vpc_auto_assigned_ipv6_association_id" {
  value = "${aws_vpc.default.ipv6_association_id}"
}

output "vpc_auto_assigned_ipv6_cidr_block" {
  value = "${aws_vpc.default.ipv6_cidr_block}"
}
