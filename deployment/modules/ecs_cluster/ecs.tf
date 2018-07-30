resource "aws_ecs_cluster" "default" {
  name = "${var.resource_prefix}-cluster"
}
