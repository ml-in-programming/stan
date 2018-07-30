output "ecs_cluster_id" {
  value = "${aws_ecs_cluster.default.id}"
}

output "ecs_cluster_arn" {
  value = "${aws_ecs_cluster.default.arn}"
}

output "iam_instance_profile_arn" {
  value = "${aws_iam_instance_profile.instance_profile.arn}"
}

output "iam_ec2_role_name" {
  value = "${aws_iam_role.ec2_instance_role.name}"
}
