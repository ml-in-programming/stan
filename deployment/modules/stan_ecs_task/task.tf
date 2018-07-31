resource "aws_ecs_task_definition" "default" {
  family = "${var.resource_prefix}-${var.internal_prefix}-task"
  task_role_arn = "${aws_iam_role.ecs_task_role.arn}"

  container_definitions = <<DEFINITION
[
    {
      "volumesFrom": [],
      "memory": 7900,
      "extraHosts": null,
      "dnsServers": null,
      "disableNetworking": null,
      "dnsSearchDomains": null,
      "portMappings": [
        {
          "hostPort": 80,
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "hostname": null,
      "essential": true,
      "entryPoint": null,
      "mountPoints": [],
      "name": "${var.resource_prefix}-${var.internal_prefix}-task",
      "ulimits": [
        {
          "softLimit": 50000,
          "hardLimit": 65535,
          "name": "nofile"
        },
        {
          "softLimit": 50000,
          "hardLimit": 65535,
          "name": "nproc"
        }
      ]
      ,
      "dockerSecurityOptions": null,
      "environment": [],
      "links": null,
      "workingDirectory": null,
      "readonlyRootFilesystem": null,
      "image": "egorbb/stan",
      "command": null,
      "user": null,
      "dockerLabels": null,
      "logConfiguration" : {
        "logDriver": "awslogs",
        "options": {
            "awslogs-group": "${aws_cloudwatch_log_group.ecs_logs.name}",
            "awslogs-region": "${var.aws_region}",
            "awslogs-stream-prefix": "${var.resource_prefix}-${var.internal_prefix}-stream"
        }
      },
      "cpu": 2048,
      "privileged": null,
      "memoryReservation": null
    }
  ]
DEFINITION
}