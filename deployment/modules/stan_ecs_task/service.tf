data "aws_ecs_task_definition" "default" {
  task_definition = "${aws_ecs_task_definition.default.family}"
  depends_on = [
    "aws_ecs_task_definition.default"]
}

resource "aws_ecs_service" "default" {
  name = "${var.resource_prefix}-${var.internal_prefix}-proxy-service"
  cluster = "${var.ecs_cluster_id}"
  desired_count = "1"
  iam_role = "${aws_iam_role.ecs_service_role.arn}"
  deployment_minimum_healthy_percent = 50
  deployment_maximum_percent = 200
  placement_constraints {
    type = "distinctInstance"
  }
  load_balancer {
    container_name = "${var.resource_prefix}-${var.internal_prefix}-task"
    container_port = "8000"
    target_group_arn = "${aws_alb_target_group.default.arn}"
  }

  task_definition = "${aws_ecs_task_definition.default.family}:${max("${aws_ecs_task_definition.default.revision}", "${data.aws_ecs_task_definition.default.revision}")}"

}