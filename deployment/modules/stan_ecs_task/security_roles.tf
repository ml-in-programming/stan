resource "aws_iam_role" "ecs_task_role" {
  assume_role_policy = "${data.aws_iam_policy_document.role_lambda_service.json}"
  name = "${var.resource_prefix}-${var.internal_prefix}-task-role"
}

data "aws_iam_policy_document" "role_lambda_service" {
  statement {
    effect = "Allow"
    principals {
      identifiers = [
        "ecs-tasks.amazonaws.com"]
      type = "Service"
    }
    actions = [
      "sts:AssumeRole"]
  }
}

resource "aws_iam_role_policy_attachment" "task_attachment_1" {
  role = "${aws_iam_role.ecs_task_role.name}"
  policy_arn = "${aws_iam_policy.ecs_task_working_policy.arn}"
}

resource "aws_iam_policy" "ecs_task_working_policy" {
  name = "${var.resource_prefix}-${var.internal_prefix}-working-policy"
  policy = "${data.aws_iam_policy_document.policy_cloudwatch_logs_json.json}"
}

data "aws_iam_policy_document" "policy_cloudwatch_logs_json" {
  statement {
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams"
    ]
    resources = [
      "arn:aws:logs:*"]
  }

  statement {
    effect = "Allow"
    actions = [
      "es:*"
    ]
    resources = [
      "*"
    ]
  }
}


resource "aws_iam_role" "ecs_service_role" {
  assume_role_policy = "${data.aws_iam_policy_document.ecs_service_assume_role_json.json}"
  name = "${var.resource_prefix}-${var.internal_prefix}-service-role"
}

data "aws_iam_policy_document" "ecs_service_assume_role_json" {
  "statement" {
    effect = "Allow"
    principals {
      identifiers = [
        "ecs.amazonaws.com"]
      type = "Service"
    }
    actions = [
      "sts:AssumeRole"]
  }
}


resource "aws_iam_role_policy_attachment" "service_role" {
  role = "${aws_iam_role.ecs_service_role.name}"
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceRole"
}



