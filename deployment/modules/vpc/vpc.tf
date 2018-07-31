resource "aws_vpc" "default" {
  cidr_block = "10.10.0.0/16"
  assign_generated_ipv6_cidr_block = true
  enable_dns_support = true
  enable_dns_hostnames = true
}

data "aws_availability_zones" "available" {}

resource "aws_internet_gateway" "default" {
  vpc_id = "${aws_vpc.default.id}"
}

resource "aws_subnet" "private" {
  count = "${var.az_count}"
  cidr_block = "${cidrsubnet(aws_vpc.default.cidr_block, 4, count.index)}"
  availability_zone = "${data.aws_availability_zones.available.names[count.index]}"
  vpc_id = "${aws_vpc.default.id}"
}

resource "aws_subnet" "public" {
  count = "${var.az_count}"
  cidr_block = "${cidrsubnet(aws_vpc.default.cidr_block, 4, var.az_count+count.index)}"
  availability_zone = "${data.aws_availability_zones.available.names[count.index]}"
  vpc_id = "${aws_vpc.default.id}"
  ipv6_cidr_block = "${cidrsubnet(aws_vpc.default.ipv6_cidr_block, 8, 2+count.index)}"
  assign_ipv6_address_on_creation = false
}


resource "aws_eip" "nat" {
  count = "${var.az_count}"
}

resource "aws_nat_gateway" "gw" {
  count = "${var.az_count}"
  allocation_id = "${aws_eip.nat.*.id[count.index]}"
  subnet_id = "${aws_subnet.public.*.id[count.index]}"
}

resource "aws_route_table" "public" {
  vpc_id = "${aws_vpc.default.id}"

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = "${aws_internet_gateway.default.id}"
  }

  route {
    ipv6_cidr_block = "::/0"
    gateway_id = "${aws_internet_gateway.default.id}"
  }
}

resource "aws_route_table_association" "public" {
  count = "${var.az_count}"
  subnet_id = "${aws_subnet.public.*.id[count.index]}"
  route_table_id = "${aws_route_table.public.id}"
}

resource "aws_route_table" "private" {
  count = "${var.az_count}"
  vpc_id = "${aws_vpc.default.id}"

  route {
    cidr_block = "0.0.0.0/0"
    nat_gateway_id = "${aws_nat_gateway.gw.*.id[count.index]}"
  }
}

resource "aws_route_table_association" "private" {
  count = "${var.az_count}"
  subnet_id = "${aws_subnet.private.*.id[count.index]}"
  route_table_id = "${aws_route_table.private.*.id[count.index]}"
}
