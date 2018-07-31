resource "aws_iam_role" "ec2_instance_role" {
  assume_role_policy = "${data.aws_iam_policy_document.ec2_instance_assume_policy.json}"
  name = "${var.resource_prefix}-ec2-instance-role"
}

data "aws_iam_policy_document" "ec2_instance_assume_policy" {
  "statement" {
    effect = "Allow"
    principals {
      identifiers = [
        "ec2.amazonaws.com"]
      type = "Service"
    }
    actions = [
      "sts:AssumeRole"]
  }
}


resource "aws_iam_instance_profile" "instance_profile" {
  name = "${var.resource_prefix}-ec2-instance-profile"
  role = "${aws_iam_role.ec2_instance_role.name}"
}

resource "aws_iam_role_policy_attachment" "instance_role" {
  role = "${aws_iam_role.ec2_instance_role.name}"
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}



