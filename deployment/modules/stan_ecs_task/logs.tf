resource "aws_cloudwatch_log_group" "ecs_logs" {
  name = "${var.resource_prefix}-${var.internal_prefix}-log-group"
  retention_in_days = 7
}